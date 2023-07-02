import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

"""
Scenario 5: two GPUs 
    (1) partitions the model
    (2) run through an arbitrary single-machine model, and then 
    (3) updates the loss incorporating the result on both GPU (using all-reduce on weight)

The input data 10 x 10. The model is 10 x 10, partitioned among two GPUs, so
10 x 5 in each. The output data is 10 x 10. 

Note:
1. all_gather doesn't preserve gradient, so we need to override the local output
   more information: https://discuss.pytorch.org/t/dist-all-gather-and-gradient-preservation-in-multi-gpu-training/120696
2. after overriding the local output, we need to scale the loss back by world_size
   more information: https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
"""

def process(rank, world_size, data, labels):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    model = nn.Linear(10, 5).to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    print(f"rank{rank}: model has weights\n{model.weight.data}\n")

    # forward pass
    output_local = model(data.to(rank))
    output_list = [torch.empty_like(output_local) for _ in range(dist.get_world_size())]
    dist.all_gather(output_list, output_local, group=group)
    output_list[rank] = output_local # override since all_gather doesn't preserve gradient
    outputs = torch.cat(output_list, 1)
    print(f"rank{rank}: output is {outputs}\n")

    # backward pass
    labels = labels.to(rank)
    loss = world_size * loss_fn(outputs, labels) # after override, we need to scale the gradient back
    loss.backward()

    # update parameters
    optimizer.step()
    print(f"rank{rank}: model now has weights\n{model.weight.data}\n")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    data = torch.randn(10, 10)
    labels = torch.randn(10, 10)
    mp.spawn(process, args=(world_size, data, labels), nprocs=world_size, join=True)
