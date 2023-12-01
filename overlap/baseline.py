import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os

NNODES = 1               # Number of nodes
MODEL_PARALLEL_SIZE = 2  # Number of GPUs per node
RUN_WITH_CPU = True      # Run with CPU instead of GPU
B, L, H = 24, 1024, 2560 # Batch size, sequence length, hidden size
NUM_TILES = 1            # Number of tiles to split the input into

_MODEL_PARALLEL_GROUP = None

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

    def forward(self, x):
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        return x


def parallel_matmul_and_reduce(input_, model, num_tiles):
    if num_tiles == 1:
        time_0 = time.time()
        output_ = model(input_)
        time_1 = time.time()
        dist.all_reduce(output_, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP)
        time_2 = time.time()
        print(f"Rank {dist.get_rank()}: Time for matmul: {time_1 - time_0:.04f} seconds")
        print(f"Rank {dist.get_rank()}: Time for all-reduce: {time_2 - time_1:.04f} seconds")
        return output_

    input_splits = torch.chunk(input_, num_tiles, dim=0)
    output_ = torch.zeros_like(input_)
    handles = []
    
    # Launch non-blocking all-reduce operations
    for i, input_part in enumerate(input_splits):
        output_part = model(input_part)
        handle = dist.all_reduce(output_part, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP, async_op=True)
        handles.append((handle, output_part, i))

    # Wait for all operations to complete and gather results
    for handle, c_part, i in handles:
        handle.wait()  # Wait for the all-reduce to complete
        output_[i * c_part.shape[0]:(i + 1) * output_part.shape[0], :] = output_part

    return output_


def run(rank, world_size):
    parallel_init(rank, world_size)
    assert _MODEL_PARALLEL_GROUP is not None

    # Create the MLP model
    mlp = TwoLayerMLP(H, 4 * H // MODEL_PARALLEL_SIZE)

    # Create tensors
    input_ = torch.randn(B, L, H)  # Adjust input dimensions

    # using tiles
    start_tiled_time = time.time()
    output_ = parallel_matmul_and_reduce(input_, mlp, num_tiles=NUM_TILES)
    end_tiled_time = time.time()
    print(f"Rank {rank}: Time for tiled matrix multiplication: {end_tiled_time - start_tiled_time:.04f} seconds")


if __name__ == "__main__":
    print(f"Running on {MODEL_PARALLEL_SIZE} GPUs per node, tile size {NUM_TILES}")
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)