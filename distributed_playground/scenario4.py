import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Scenario 4: two GPUs 
#   (1) partitions input, 
#   (2) run through an arbitrary single-machine model, and then 
#   (3) updates the loss incorporating the result on both GPU (using all-reduce on weight)


# Any custom model, let's use a basic one
MODEL = nn.Linear(5, 5)

def process(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = MODEL.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(f"rank{rank}: model has weights\n{ddp_model.module.weight.data}\n")

    # forward pass
    outputs = ddp_model(torch.randn(20, 5).to(rank))
    labels = torch.randn(20, 5).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print(f"rank{rank}: model now has weights\n{ddp_model.module.weight.data}\n")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(process, args=(world_size,), nprocs=world_size, join=True)
