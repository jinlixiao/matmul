import argparse
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import utils

NNODES = 1                 # Number of nodes
MODEL_PARALLEL_SIZE = 2    # Number of GPUs per node
RUN_WITH_CPU = False       # Run with CPU instead of GPU
B, L, H = 24, 1024, 2560   # Batch size, sequence length, hidden size
EXCLUDE_ITERATIONS = 3     # Number of iterations to exclude from statistics

parser = argparse.ArgumentParser(description='Tiled Matrix Multiplication')
parser.add_argument('--num_tiles', type=int, default=1, help='Number of tiles to split the input into')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations to run')
args = parser.parse_args()

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoLayerMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x, num_tiles):
        # Split among the batch dimension
        input_splits = torch.chunk(x, num_tiles, dim=0)
        output_list = []
        total_duration = 0

        # Launch non-blocking all-gather operations
        for i, input_part in enumerate(input_splits):
            output_part, duration = self.single_forward(input_part)
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

    def single_forward(self, x):
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

def run(rank, world_size):
    utils.parallel_init(rank, world_size, run_with_cpu=RUN_WITH_CPU)
    model_parallel_group = utils.get_model_parallel_group()
    assert model_parallel_group is not None

    # Create the MLP model
    if RUN_WITH_CPU:
        mlp = TwoLayerMLP(H, 4 * H // MODEL_PARALLEL_SIZE)
        input_ = torch.randn(B, L, H)
    else:
        mlp = TwoLayerMLP(H, 4 * H // MODEL_PARALLEL_SIZE).cuda(rank)
        input_ = torch.randn(B, L, H).cuda(rank)

    # Run the model
    running_time = []
    with torch.no_grad():
        for i in range(args.num_iterations + EXCLUDE_ITERATIONS):
            utils.print_rank_0(f"********** Iteration {i} **********")
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)

            start_event.record()  # Start timing
            input_ = mlp(input_, num_tiles=args.num_tiles)
            end_event.record()  # End timing
            cuda.synchronize()  # Wait for the events to be recorded
            
            duration = start_event.elapsed_time(end_event)
            if i >= EXCLUDE_ITERATIONS:
                running_time.append(duration)
            utils.print_rank_0(f"Rank {rank}: Time for tiled matrix multiplication: {duration:.04f} milliseconds")

    # Print statistics
    utils.print_rank_0(f"\n********** Statistics **********")
    utils.print_rank_0(f"Average time for tiled matrix multiplication: {sum(running_time) / len(running_time):.04f} milliseconds")
    utils.print_rank_0(f"Median time for tiled matrix multiplication: {sorted(running_time)[len(running_time) // 2]:.04f} milliseconds")

if __name__ == "__main__":
    print(f"Running on {MODEL_PARALLEL_SIZE} GPUs per node, tile size {args.num_tiles}")
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)