import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import utils

_MODEL_REGISTRY = {}

def get_model_class(model_name):
    try:
        return _MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}")
    
def register_model(cls):
    _MODEL_REGISTRY[cls.__name__] = cls
    return cls

@register_model
class OverlapTiledAllreduceMLP(nn.Module):
    """
    Overlapped version of the two-layer MLP model.

    GEMM + ReLU + GEMM + All-reduce

    Use tiling to overlap (GEMM + ReLU + GEMM) and (All-reduce)
    """

    def __init__(self, input_size, hidden_size):
        super(OverlapTiledAllreduceMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x, num_tiles = 1):
        # Split among the batch dimension
        input_splits = torch.chunk(x, num_tiles, dim=0)
        output_ = torch.zeros_like(x)
        handles = []

        total_duration = 0

        # Launch non-blocking all-reduce operations
        for i, input_part in enumerate(input_splits):
            output_part, duration = self._single_forward(input_part)
            handle = dist.all_reduce(output_part, op=dist.ReduceOp.SUM, group=utils.get_model_parallel_group(), async_op=True)
            handles.append((handle, output_part, i))
            total_duration += duration

        # Wait for all operations to complete and gather results
        for handle, output_part, i in handles:
            handle.wait()
            output_[i * output_part.shape[0]:(i + 1) * output_part.shape[0], :] = output_part

        utils.print_rank_0(f"Rank {dist.get_rank()}: Total time for single_forward: {total_duration:.04f} milliseconds")
        return output_

    def _single_forward(self, x):
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        end_event.record()  # End timing

        cuda.synchronize()  # Wait for the events to be recorded

        duration = start_event.elapsed_time(end_event)
        utils.print_rank_0(f"Rank {dist.get_rank()}: single_forward duration: {duration:.04f} milliseconds")
        return x, duration

@register_model
class NonOverlapTiledAllreduceMLP(nn.Module):
    """
    Non-overlapped version of the two-layer MLP model.

    GEMM + ReLU + GEMM + All-reduce
    """

    def __init__(self, input_size, hidden_size):
        super(NonOverlapTiledAllreduceMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x, num_tiles = 1):
        # Split among the batch dimension
        input_splits = torch.chunk(x, num_tiles, dim=0)
        output_ = torch.zeros_like(x)

        total_duration = 0
        all_reduce_duration = 0

        for i, input_part in enumerate(input_splits):
            output_part, duration = self._single_forward(input_part)
            total_duration += duration
            ar_start_event = cuda.Event(enable_timing=True)
            ar_end_event = cuda.Event(enable_timing=True)
            
            ar_start_event.record()
            dist.all_reduce(output_part, op=dist.ReduceOp.SUM, group=utils.get_model_parallel_group())
            ar_end_event.record()
            output_[i * output_part.shape[0]:(i + 1) * output_part.shape[0], :] = output_part
            
            # Calculate all_reduce duration
            cuda.synchronize()
            ar_duration = ar_start_event.elapsed_time(ar_end_event)
            utils.print_rank_0(f"Rank {dist.get_rank()}: All-reduce duration for tile {i}: {ar_duration:.04f} milliseconds")
            all_reduce_duration += ar_duration

        utils.print_rank_0(f"Rank {dist.get_rank()}: Total computation time: {total_duration:.04f} milliseconds")
        utils.print_rank_0(f"Rank {dist.get_rank()}: Total communication time: {all_reduce_duration:.04f} milliseconds")
        return output_

    def _single_forward(self, x):
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        end_event.record()  # End timing

        cuda.synchronize()  # Wait for the events to be recorded

        duration = start_event.elapsed_time(end_event)
        utils.print_rank_0(f"Rank {dist.get_rank()}: single_forward duration: {duration:.04f} milliseconds")
        return x, duration

@register_model
class OverlapTiledAllGatherMLP(nn.Module):
    """
    Overlapped version of the two-layer MLP model.

    (GEMM + ReLU + GEMM) overlapping with (All-gather)

    Purpose is to experiment whether all-gather gives the same slowdown
    to the computation as all-reduce.
    """

    def __init__(self, input_size, hidden_size):
        super(OverlapTiledAllGatherMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x, num_tiles = 1):
        # Split among the batch dimension
        input_splits = torch.chunk(x, num_tiles, dim=0)
        output_list = []
        total_duration = 0

        # Launch non-blocking all-gather operations
        for i, input_part in enumerate(input_splits):
            output_part, duration = self._single_forward(input_part)
            gathered_outputs = [torch.zeros_like(output_part) for _ in range(dist.get_world_size())]
            handle = dist.all_gather(gathered_outputs, output_part, group=utils.get_model_parallel_group(), async_op=True)
            output_list.append((handle, gathered_outputs))
            total_duration += duration

        # Wait for all operations to complete
        for handle, _ in output_list:
            handle.wait()

        # Correctly combine the gathered results
        outputs_combined = []
        for _, gathered in output_list:
            for tensor in gathered:
                outputs_combined.append(tensor)

        output_ = torch.cat(outputs_combined, dim=0)

        utils.print_rank_0(f"Rank {dist.get_rank()}: Total time for single_forward: {total_duration:.04f} milliseconds")
        return x

    def _single_forward(self, x):
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        end_event.record()  # End timing

        cuda.synchronize()  # Wait for the events to be recorded

        duration = start_event.elapsed_time(end_event)
        utils.print_rank_0(f"Rank {dist.get_rank()}: single_forward duration: {duration:.04f} milliseconds")
        return x, duration
    
@register_model
class NonOverlapNontiledMLP(nn.Module):
    """
    Non-overlapped version of the two-layer MLP model.

    GEMM + ReLU + GEMM + Reduce-scatter + All-gather
    """

    def __init__(self, input_size, hidden_size):
        super(NonOverlapNontiledMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        # Start timing for the forward pass
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        start_event.record()

        # First layer operations
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)

        # Second layer operations
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias

        # End timing for matrix multiplication
        end_event.record()
        cuda.synchronize()
        single_forward_duration = start_event.elapsed_time(end_event)

        # Determine the size for each split
        world_size = dist.get_world_size()
        split_size = x.size(0) // world_size
        assert x.size(0) % world_size == 0

        # Split the output for reduce-scatter operation
        outputs_split = list(torch.split(x, split_size, dim=0))

        # Record time for reduce-scatter operation
        start_event.record()
        scatter_output = torch.zeros_like(outputs_split[dist.get_rank()])
        dist.reduce_scatter(scatter_output, outputs_split, group=utils.get_model_parallel_group())
        end_event.record()

        # Wait for the event to be recorded and calculate duration
        cuda.synchronize()
        reduce_scatter_duration = start_event.elapsed_time(end_event)

        # All-gather operation
        start_event.record()
        gather_list = [torch.zeros_like(scatter_output) for _ in range(world_size)]
        dist.all_gather(gather_list, scatter_output, group=utils.get_model_parallel_group())
        end_event.record()

        # Wait for the event to be recorded and calculate duration
        cuda.synchronize()
        all_gather_duration = start_event.elapsed_time(end_event)

        # Concatenate the gathered outputs
        output = torch.cat(gather_list, dim=0)

        utils.print_rank_0(f"Rank {dist.get_rank()}: Total time for matrix multiplications: {single_forward_duration:.04f} milliseconds")
        utils.print_rank_0(f"Rank {dist.get_rank()}: Total time for reduce_scatter: {reduce_scatter_duration:.04f} milliseconds")
        utils.print_rank_0(f"Rank {dist.get_rank()}: Total time for all_gather: {all_gather_duration:.04f} milliseconds")
        return output


@register_model
class OverlapNontiledMLP(nn.Module):
    """
    Overlapped version of the two-layer MLP model.

    GEMM + ReLU + GEMM + Reduce-scatter + All-gather

    The second GEMM overlaps with Reduce-scatter
    """

    def __init__(self, input_size, hidden_size):
        super(OverlapNontiledMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # First GEMM
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)

        # Prepare for second GEMM with overlap
        split_size = self.fc2_weight.size(0) // world_size
        assert self.fc2_weight.size(0) % world_size == 0
        B_chunks = torch.chunk(self.fc2_weight, world_size, dim=0)

        send_result = torch.zeros(x.size(0), x.size(1), split_size, device=rank)
        recv_result = torch.zeros(x.size(0), x.size(1), split_size, device=rank)

        # Overlapping second GEMM with ReduceScatter
        for i in range(world_size):

            # Non-blocking send and receive
            send_rank = (rank - 1 + world_size) % world_size
            recv_rank = (rank + 1) % world_size

            send_op = dist.P2POp(dist.isend, send_result, send_rank)
            recv_op = dist.P2POp(dist.irecv, recv_result, recv_rank)
            reqs = dist.batch_isend_irecv([send_op, recv_op])

            # Compute A * Bi
            chunk_index = (rank + i + 1) % world_size
            chunk_result = torch.matmul(x, B_chunks[chunk_index].t())

            for req in reqs:
                req.wait()
            send_result = recv_result + chunk_result

        # All Gather
        gather_list = [torch.zeros_like(send_result) for _ in range(world_size)]
        dist.all_gather(gather_list, send_result, group=utils.get_model_parallel_group())
        output = torch.cat(gather_list, dim=2)
        return output