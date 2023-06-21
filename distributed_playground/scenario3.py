#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Scenario 3: two CPUs (GPUs)
#   (1) partition the input/weight, 
#   (2) compute matmul, and 
#   (3) allgather to get the same result

def get_data(rank):
    """
    Just partition the data to different GPUs.
    """
    world_size = dist.get_world_size()
    data = torch.tensor([[1, 2, 3, 4, 5, 6]]).t()
    size = data.size(0) // world_size
    return data[size * rank: size * (rank + 1), :]

def process(rank, size):
    # create default process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    group = dist.new_group([0, 1])

    # create local model
    tensor = get_data(rank)
    weight = torch.tensor([[1, 2]])
    result = torch.matmul(tensor, weight)
    print(f"rank{rank}: partial result is\n{result}\n")

    # all_gather
    tensor_list = [torch.empty_like(result) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, result, group=group)
    full_result = torch.cat(tensor_list, 0)
    print(f"rank{rank}: full result is\n{full_result}\n")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(process, args=(world_size,), nprocs=world_size, join=True)
