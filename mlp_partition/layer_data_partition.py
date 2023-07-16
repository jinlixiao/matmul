import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from layers import DataParallelLinearModel
from partition import partition_tensor

"""
One Layer MLP Data Parallel
"""

def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # setup model
    data = partition_tensor(data, world_size, rank, dim=0)
    labels = partition_tensor(labels, world_size, rank, dim=0)
    model = DataParallelLinearModel(weights, group)
    model = model.to(rank)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

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
    # communication configurations
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # model configurations
    world_size = 2
    torch.manual_seed(0)
    data = torch.randn(4, 4)
    labels = torch.randn(4, 4)
    weights = torch.randn(4, 4)
    
    # run
    start_time = time.time()
    mp.spawn(process, args=(world_size, data, labels, weights), nprocs=world_size, join=True)
    print(f"total time: {time.time() - start_time:.3f} seconds")

