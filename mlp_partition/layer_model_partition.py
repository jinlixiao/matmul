import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from layers import ModelParallelLinearModel

"""
One Layer MLP Model Parallel
"""

def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # todo: rewrite data spliting in a more general way
    data_size = data.size(1) // dist.get_world_size()
    weights = weights[:, rank * data_size : (rank + 1) * data_size]

    # model setup
    model = ModelParallelLinearModel(weights, group)
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
    torch.manual_seed(0)
    data = torch.randn(4, 4)
    labels = torch.randn(4, 4)
    weights = torch.randn(4, 4)
    start_time = time.time()
    mp.spawn(process, args=(world_size, data, labels, weights), nprocs=world_size, join=True)
    print(f"total time: {time.time() - start_time:.3f} seconds")
