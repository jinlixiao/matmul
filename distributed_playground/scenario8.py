import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

"""
Scenario 8: two GPUs 
    (1) partitions the model
    (2) run through an arbitrary single-machine model, and then 
    (3) updates the loss incorporating the result on both GPU (using all-reduce on weight)

The input data 10 x 10. The model is 10 x 10, partitioned among two GPUs, so
10 x 5 in each. The output data is 10 x 10. 
"""


class LinearPartitionedModel(nn.Module):

    def __init__(self, weights, group):
        super(LinearPartitionedModel, self).__init__()
        rank = dist.get_rank()

        data_size = weights.size(0) // dist.get_world_size()
        weights_local = weights[:, rank * data_size : (rank + 1) * data_size]

        self.weights = nn.Parameter(torch.empty_like(weights_local))
        self.weights.data = weights_local.t().to(dist.get_rank())
        self.group = group

    def forward(self, data):
        output_local = data.to(dist.get_rank()).mm(self.weights.t())
        output_list = [torch.empty_like(output_local) for _ in range(dist.get_world_size())]
        dist.all_gather(output_list, output_local, group=self.group)
        output_list[dist.get_rank()] = output_local
        output = torch.cat(output_list, dim=1).contiguous()
        return output


def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    model = LinearPartitionedModel(weights, group)
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # forward pass
    outputs = model(data)

    # backward pass
    labels = labels.to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()

    # update parameters
    optimizer.step()
    torch.set_printoptions(sci_mode=False, precision=4)
    print(f"rank{rank}: model now has weights\n{model.weights.data}\n")


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
