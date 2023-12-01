import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

NNODES = 1               # Number of nodes
MODEL_PARALLEL_SIZE = 2  # Number of GPUs per node
RUN_WITH_CPU = True      # Run with CPU instead of GPU

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


def parallel_matmul_and_reduce(a, b, num_tiles):
    if num_tiles == 1:
        c = torch.matmul(a, b)
        dist.all_reduce(c, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP)
        return c

    a_splits = torch.chunk(a, num_tiles, dim=0)
    c = torch.zeros_like(a)
    handles = []
    
    # Launch non-blocking all-reduce operations
    for i, a_part in enumerate(a_splits):
        c_part = torch.matmul(a_part, b)
        handle = dist.all_reduce(c_part, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP, async_op=True)
        handles.append((handle, c_part, i))

    # Wait for all operations to complete and gather results
    for handle, c_part, i in handles:
        handle.wait()  # Wait for the all-reduce to complete
        c[i * c_part.shape[0]:(i + 1) * c_part.shape[0], :] = c_part

    return c


def run(rank, world_size):
    parallel_init(rank, world_size)
    assert _MODEL_PARALLEL_GROUP is not None

    # Create tensors
    input_ = torch.randn(1000, 100)
    weight_1 = torch.randn(100, 100)

    # using tiles
    start_tiled_time = time.time()
    c = parallel_matmul_and_reduce(input_, weight_1, num_tiles=10)
    end_tiled_time = time.time()
    print(f"Rank {rank}: Time for tiled matrix multiplication: {end_tiled_time - start_tiled_time:.04f} seconds")


if __name__ == "__main__":
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)