import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from layers import ModelParallelLinearLayer
from partition import partition_tensor

"""
One Layer MLP Model Parallel
"""

SEED = 0
INPUT_SIZE = 4
BATCH_SIZE = 4
OUTPUT_SIZE = 4
HIDDEN_SIZE = 4

def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    weights = partition_tensor(weights, world_size, rank, dim=1)
    model = ModelParallelLinearLayer(weights, group)
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # forward pass
    outputs = model(data.to(rank))

    # backward pass
    labels = labels.to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()

    # update parameters
    optimizer.step()
    torch.set_printoptions(sci_mode=False, precision=4)
    print(f"rank{rank}: model now has weights\n{model.linear.weight.data}\n")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    torch.manual_seed(SEED)
    data = torch.randn(BATCH_SIZE, INPUT_SIZE)
    labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)
    weights = torch.randn(INPUT_SIZE, OUTPUT_SIZE)
    start_time = time.time()
    mp.spawn(process, args=(world_size, data, labels, weights), nprocs=world_size, join=True)
    print(f"total time: {time.time() - start_time:.3f} seconds")
