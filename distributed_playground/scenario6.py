import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

"""
Scenario 6: two GPUs 
   (1) partitions input, 
   (2) run through a linear model, and then 
   (3) updates the loss incorporating the result on both GPU (using all-reduce on weight)

The input data 10 x 10, partitioned among two GPUs, so 5 x 10 each. 
The model is 10 x 10. 
The output data is 10 x 10, partitioned among two GPUs, so 5 x 10 each. 

Note: this is the same as scenario 4, but without using the DDP wrapper and its
      associated optimization.
"""


def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    model = nn.Linear(10, 10, bias=False)
    model.weight.data = weights.t()
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # forward pass
    data_size = data.size(0) // world_size
    data_local = data[rank * data_size : (rank + 1) * data_size].to(rank)
    outputs = model(data_local)

    # backward pass
    label_local = labels[rank * data_size : (rank + 1) * data_size].to(rank)
    loss = loss_fn(outputs, label_local)
    loss.backward()
    dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM, group=group)

    # update parameters
    optimizer.step()
    torch.set_printoptions(sci_mode=False, precision=4)
    # print(f"rank{rank}: model now has weights\n{model.weight.data}\n")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    torch.manual_seed(0)
    data = torch.randn(10, 10)
    labels = torch.randn(10, 10)
    weights = torch.randn(10, 10)
    start_time = time.time()
    mp.spawn(process, args=(world_size, data, labels, weights), nprocs=world_size, join=True)
    print(f"total time: {time.time() - start_time:.3f} seconds")
