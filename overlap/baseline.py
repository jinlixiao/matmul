import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

NNODES = 1               # Number of nodes
MODEL_PARALLEL_SIZE = 2  # Number of GPUs per node
RUN_WITH_CPU = True      # Run with CPU instead of GPU

_MODEL_PARALLEL_GROUP = None
_WORLD_SIZE = None

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

    global _WORLD_SIZE
    _WORLD_SIZE = world_size


def run(rank, world_size):
    parallel_init(rank, world_size)
    assert _MODEL_PARALLEL_GROUP is not None

    # Create tensors
    a = torch.randn(100, 100)
    b = torch.randn(100, 100)

    # matmul
    start_time_matmul = time.time()
    c = torch.matmul(a, b)
    end_time_matmul = time.time()
    time_matmul = end_time_matmul - start_time_matmul

    # all-reduce
    start_time_allreduce = time.time()
    dist.all_reduce(c, op=dist.ReduceOp.SUM, group=_MODEL_PARALLEL_GROUP)
    end_time_allreduce = time.time()
    time_allreduce = end_time_allreduce - start_time_allreduce

    # Print result and timing
    print(f"Rank {rank}: Time for matrix multiplication: {time_matmul} seconds")
    print(f"Rank {rank}: Time for all-reduce operation: {time_allreduce} seconds")


if __name__ == "__main__":
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)