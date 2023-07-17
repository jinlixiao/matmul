import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import TwoLayerDataPartitionModel
from partition import partition_tensor

"""
Two Layer MLP Data Parallel
"""

SEED = 0
INPUT_SIZE = 4
BATCH_SIZE = 4
OUTPUT_SIZE = 4
HIDDEN_SIZE = 4

def process(rank, world_size, data, labels, weights1, weights2):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # setup model
    data = partition_tensor(data, world_size, rank, dim=0)
    labels = partition_tensor(labels, world_size, rank, dim=0)
    model = TwoLayerDataPartitionModel(weights1, weights2, group)
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
    print(f"model now has weights1\n{model.layer1.linear.weight.data}\n")
    print(f"model now has weights2\n{model.layer2.linear.weight.data}\n")

if __name__ == "__main__":
    # communication configurations
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "64758"

    # model configurations
    world_size = 2
    torch.manual_seed(SEED)
    data = torch.randn(BATCH_SIZE, INPUT_SIZE)
    labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)
    weights1 = torch.randn(INPUT_SIZE, HIDDEN_SIZE)
    weights2 = torch.randn(HIDDEN_SIZE, OUTPUT_SIZE)
    
    # run
    start_time = time.time()
    mp.spawn(process, args=(world_size, data, labels, weights1, weights2), nprocs=world_size, join=True)
    print(f"total time: {time.time() - start_time:.3f} seconds")

