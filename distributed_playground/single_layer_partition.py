import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
In this snippet, we present two partitioning strategies for a single linear 
layer model:
    (1) partition the model
    (2) partition the batch
"""

INPUT_SIZE = 10
OUTPUT_SIZE = 10
HIDDEN_SIZE = 10
BATCH_SIZE = 10
DATA_SIZE = 1000

class RandomDataset(Dataset):
    def __init__(self, data_size, input_size, output_size):
        self.data_size = data_size
        self.input_size = input_size
        self.output_size = output_size
        self.data_set = torch.randn(self.data_size, self.input_size)
        self.labels = torch.randn(self.data_size, self.output_size)
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        inputs = self.data_set[idx]
        output = self.labels[idx]
        return inputs, output

def run_model_partition(rank, world_size, dataset, weights):
    """
    Partition among the model layers. We split the hidden layer dimension into
    world_size GPUs. 
    """
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    hidden_size = HIDDEN_SIZE // world_size
    model = nn.Linear(10, 5, bias=False)
    model.weight.data = weights[:, rank * hidden_size : (rank + 1) * hidden_size].t()
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # start training
    for data, labels in dataloader:
        optimizer.zero_grad()

        # forward pass
        output_local = model(data.to(rank))
        output_list = [torch.empty_like(output_local) for _ in range(dist.get_world_size())]
        dist.all_gather(output_list, output_local, group=group)
        output_list[rank] = output_local  # override since all_gather doesn't preserve gradient
        outputs = torch.cat(output_list, 1)

        # backward pass
        labels = labels.to(rank)
        loss = world_size * loss_fn(outputs, labels)  # after override, we need to scale the gradient back
        loss.backward()

        # update parameters
        optimizer.step()

    # inspect result
    torch.set_printoptions(sci_mode=False, precision=4)
    print(f"rank{rank}: model now has weights\n{model.weight.data}\n")


def run_batch_partition(rank, world_size, dataset, weights):
    """
    Partition among the batch dimensions. We split the input data into two GPUs. 
    """
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch_size = BATCH_SIZE // world_size
    model = nn.Linear(10, 10, bias=False)
    model.weight.data = weights.t()
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for data, labels in dataloader:
        optimizer.zero_grad()

        # forward pass
        data_local = data[rank * batch_size : (rank + 1) * batch_size].to(rank)
        outputs = model(data_local)

        # backward pass
        label_local = labels[rank * batch_size : (rank + 1) * batch_size].to(rank)
        loss = loss_fn(outputs, label_local)
        loss.backward()
        dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM, group=group)

        # update parameters
        optimizer.step()

    # inspect result
    torch.set_printoptions(sci_mode=False, precision=4)
    print(f"rank{rank}: model now has weights\n{model.weight.data}\n")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    torch.manual_seed(0)
    dataset = RandomDataset(DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE)
    weights = torch.randn(10, 10)
    start_time = time.time()
    mp.spawn(run_model_partition, args=(world_size, dataset, weights), nprocs=world_size, join=True)
    mid_time = time.time()
    mp.spawn(run_batch_partition, args=(world_size, dataset, weights), nprocs=world_size, join=True)
    end_time = time.time()
    print(f"model partition takes {mid_time - start_time:.3f} seconds")
    print(f"batch partition takes {end_time - mid_time:.3f} seconds")
