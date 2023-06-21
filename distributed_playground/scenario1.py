import os
import torch.multiprocessing as mp

# Scenario 1: spawn two processes on two CPUs, each prints out a message

def process(rank):
    print(f"rank{rank}: hello world")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(process, args=(), nprocs=world_size, join=True)
