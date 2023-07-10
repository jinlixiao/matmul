import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

"""
Scenario 7: two GPUs 
   (1) partitions input, 
   (2) run through a linear model, and then 
   (3) updates the loss incorporating the result on both GPU (using all-reduce on weight)

The input data 10 x 10, partitioned among two GPUs, so 5 x 10 each. 
The model is 10 x 10. 
The output data is 10 x 10, partitioned among two GPUs, so 5 x 10 each. 

Note: this is the same as scenario 4, but without using the DDP wrapper and its
      associated optimization.
"""

class LinearSplitInBatchFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, group):
        ctx.save_for_backward(input, weight)

        # split the input data among batches
        rank = dist.get_rank()
        data_size = input.size(0) // dist.get_world_size()
        input_local = input[rank * data_size : (rank + 1) * data_size].to(rank)

        output_local = input_local.mm(weight.t())

        # gather the output data among batches
        tensor_list = [torch.empty_like(output_local) for _ in range(dist.get_world_size())]
        tensor_list[dist.get_rank()] = output_local
        dist.all_gather(tensor_list, output_local, group=group)
        output = torch.cat(tensor_list, dim=0).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input.to(dist.get_rank()))
        return grad_input, grad_weight, None


class LinearDistributedModel(nn.Module):

    def __init__(self, weights, group):
        super(LinearDistributedModel, self).__init__()
        self.weights = nn.Parameter(torch.empty_like(weights))
        self.weights.data = weights.t().to(dist.get_rank())
        self.group = group

    def forward(self, data):
        return LinearSplitInBatchFunction.apply(data, self.weights, self.group)


def process(rank, world_size, data, labels, weights):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])
    torch.set_printoptions(sci_mode=False, precision=4)

    # model setup
    model = LinearDistributedModel(weights, group)
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
