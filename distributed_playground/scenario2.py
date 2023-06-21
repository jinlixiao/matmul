import os
import torch
import torch.multiprocessing as mp

# Scenario 2: spawn two processes on two GPUs, each prints out a message

def process(rank):
    tensor = torch.tensor([rank]).to(rank)
    print(f"rank{rank}: have tensor {tensor}")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(process, args=(), nprocs=world_size, join=True)
