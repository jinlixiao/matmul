import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models
from data import RandomDataset
from partition import partition_tensor

"""
In this snippet, we present two partitioning strategies for a single linear 
layer model:
    (1) partition the model
    (2) partition the batch
"""
SEED = 0

INPUT_SIZE = 10
OUTPUT_SIZE = 10
HIDDEN_SIZE = 10
BATCH_SIZE = 10

WORLD_SIZE = 2
DATA_SIZE = 10000
EPOCHS = 10

DEBUG = False

def run_model_partition(rank, world_size, dataset, weights):
    """
    Partition among the model layers. We split the hidden layer dimension into
    world_size GPUs. 
    """
    start_time = time.time()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = models.OneLayerModelPartitionModel(weights, group)
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # start training
    for epoch in range(EPOCHS):
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(data.to(rank))
            labels = labels.to(rank)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Total Time: {time.time() - start_time:.4f}")

    # inspect result
    if DEBUG:
        torch.set_printoptions(sci_mode=False, precision=4)
        print(f"rank{rank}: model now has weights\n{model.layer.linear.weight.data}\n")


def run_batch_partition(rank, world_size, dataset, weights):
    """
    Partition among the batch dimensions. We split the input data into two GPUs. 
    """
    start_time = time.time()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = models.OneLayerDataPartitionModel(weights, group)
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # start training
    for epoch in range(EPOCHS):
        for i, (data, labels) in enumerate(dataloader):
            data = partition_tensor(data, world_size, rank, dim=0)
            labels = partition_tensor(labels, world_size, rank, dim=0)

            optimizer.zero_grad()
            outputs = model(data.to(rank))
            labels = labels.to(rank)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Total Time: {time.time() - start_time:.4f}")

    # inspect result
    if DEBUG:
        torch.set_printoptions(sci_mode=False, precision=4)
        print(f"rank{rank}: model now has weights\n{model.layer.linear.weight.data}\n")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = WORLD_SIZE

    print("Running model partition training...")
    torch.manual_seed(SEED)
    dataset = RandomDataset(DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE, seed=SEED)
    weights = torch.randn(INPUT_SIZE, OUTPUT_SIZE)
    mp.spawn(run_model_partition, args=(world_size, dataset, weights), nprocs=world_size, join=True)
    print("Finished model partition training")
    print("============================================")
    print()

    print("Running batch partition training...")
    torch.manual_seed(SEED)
    dataset = RandomDataset(DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE, seed=SEED)
    weights = torch.randn(INPUT_SIZE, OUTPUT_SIZE)
    mp.spawn(run_batch_partition, args=(world_size, dataset, weights), nprocs=world_size, join=True)
    print("Finished batch partition training")
