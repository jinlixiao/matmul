import argparse
import time
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os

NNODES = 1                 # Number of nodes
MODEL_PARALLEL_SIZE = 2    # Number of GPUs per node
RUN_WITH_CPU = False       # Run with CPU instead of GPU
B, L, H = 24, 1024, 2560   # Batch size, sequence length, hidden size
EXCLUDE_ITERATIONS = 3     # Number of iterations to exclude from statistics

parser = argparse.ArgumentParser(description='Tiled Matrix Multiplication')
parser.add_argument('--num_tiles', type=int, default=1, help='Number of tiles to split the input into')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations to run')
args = parser.parse_args()

NUM_TILES = args.num_tiles
NUM_ITER = args.num_iterations

_MODEL_PARALLEL_GROUP = None
_GLOBAL_START_TIME = None

def parallel_init(rank, world_size):
    if not RUN_WITH_CPU:
        assert torch.cuda.is_available()
        torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    if RUN_WITH_CPU:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = dist.new_group(range(world_size))

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoLayerMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x, num_tiles):
        # if num_tiles == 1:
        #     time_0 = time.time()
        #     x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        #     x = F.relu(x)
        #     x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        #     time_1 = time.time()
        #     dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP)
        #     time_2 = time.time()
        #     print(f"Rank {dist.get_rank()}: Time for matmul: {time_1 - time_0:.04f} seconds")
        #     print(f"Rank {dist.get_rank()}: Time for all-reduce: {time_2 - time_1:.04f} seconds")
        #     return x

        input_splits = torch.chunk(x, num_tiles, dim=0)
        output_ = torch.zeros_like(x)
        handles = []

        total_duration = 0

        # Launch non-blocking all-reduce operations
        for i, input_part in enumerate(input_splits):
            output_part, duration = self.single_forward(input_part)
            handle = dist.all_reduce(output_part, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP, async_op=True)
            handles.append((handle, output_part, i))
            total_duration += duration

        # Wait for all operations to complete and gather results
        for handle, output_part, i in handles:
            handle.wait()
            output_[i * output_part.shape[0]:(i + 1) * output_part.shape[0], :] = output_part

        print_rank_0(f"Rank {dist.get_rank()}: Total time for single_forward: {total_duration:.04f} milliseconds")
        return output_

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
        print_rank_0(f"Rank {dist.get_rank()}: single_forward duration: {duration:.04f} milliseconds")
        return x, duration

def run(rank, world_size):
    parallel_init(rank, world_size)
    assert _MODEL_PARALLEL_GROUP is not None

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
        for i in range(NUM_ITER + EXCLUDE_ITERATIONS):
            print_rank_0(f"********** Iteration {i} **********")
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)

            start_event.record()  # Start timing
            input_ = mlp(input_, num_tiles=NUM_TILES)
            end_event.record()  # End timing
            cuda.synchronize()  # Wait for the events to be recorded
            
            duration = start_event.elapsed_time(end_event)
            running_time.append(duration)
            print_rank_0(f"Rank {rank}: Time for tiled matrix multiplication: {duration:.04f} milliseconds")


    # Print statistics
    print_rank_0(f"\n********** Statistics **********")
    print_rank_0(f"Average time for tiled matrix multiplication: {sum(running_time) / len(running_time):.04f} milliseconds")
    print_rank_0(f"Median time for tiled matrix multiplication: {sorted(running_time)[len(running_time) // 2]:.04f} milliseconds")

def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def get_timestamp(clear=False):
    global _GLOBAL_START_TIME
    if clear or _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        return 0
    else:
        return time.time() - _GLOBAL_START_TIME

if __name__ == "__main__":
    print(f"Running on {MODEL_PARALLEL_SIZE} GPUs per node, tile size {NUM_TILES}")
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
