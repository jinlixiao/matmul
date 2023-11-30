import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    """ Distributed function to be implemented later. """

    # Set the GPU to use
    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    model_parallel_group = dist.new_group([0, 1])

    # Create tensors
    a = torch.randn(100, 100).cuda()
    b = torch.randn(100, 100).cuda()

    # Perform matrix multiplication
    c = torch.matmul(a, b)

    # All-reduce operation
    dist.all_reduce(c, op=dist.ReduceOp.SUM, group=model_parallel_group)

    # Print result
    print(f'Rank {rank}, Result: \n{c}')

if __name__ == "__main__":
    NNODES = 1               # Number of nodes
    MODEL_PARALLEL_SIZE = 2  # Number of GPUs per node
    
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
