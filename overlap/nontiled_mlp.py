import argparse
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import utils

NNODES = 1                 # Number of nodes
RUN_WITH_CPU = False       # Run with CPU instead of GPU
B, L, H = 24, 1024, 2560   # Batch size, sequence length, hidden size
EXCLUDE_ITERATIONS = 3     # Number of iterations to exclude from statistics

parser = argparse.ArgumentParser(description='Tiled Matrix Multiplication')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations to run')
parser.add_argument('--num_devices', type=int, default=torch.cuda.device_count(), help='Number of devices to use')
args = parser.parse_args()

MODEL_PARALLEL_SIZE = args.num_devices    # Number of GPUs per node

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoLayerMLP, self).__init__()
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
            input_ = mlp(input_)
            end_event.record()  # End timing
            cuda.synchronize()  # Wait for the events to be recorded
            
            duration = start_event.elapsed_time(end_event)
            if i >= EXCLUDE_ITERATIONS:
                running_time.append(duration)
            utils.print_rank_0(f"Rank {rank}: Time for nontiled matrix multiplication: {duration:.04f} milliseconds")

    # Print statistics
    utils.print_rank_0(f"\n********** Statistics **********")
    utils.print_rank_0(f"Average time for tiled matrix multiplication: {sum(running_time) / len(running_time):.04f} milliseconds")
    utils.print_rank_0(f"Median time for tiled matrix multiplication: {sorted(running_time)[len(running_time) // 2]:.04f} milliseconds")

if __name__ == "__main__":
    print(f"Running on {MODEL_PARALLEL_SIZE} GPUs per node")
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)