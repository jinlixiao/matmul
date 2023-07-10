import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

"""
Single Process Model: one CPU

The input data 10 x 10.
The model is 10 x 10. 
The output data is 10 x 10.
"""

class LinearModel(nn.Module):

    def __init__(self, weights):
        super(LinearModel, self).__init__()
        self.weights = nn.Parameter(torch.empty_like(weights))
        self.weights.data = weights.t()

    def forward(self, data):
        return data.mm(self.weights.t())


def process(data, labels, weights):
    torch.set_printoptions(sci_mode=False, precision=4)

    # model setup
    model = LinearModel(weights)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # forward pass
    outputs = model(data)

    # backward pass
    loss = loss_fn(outputs, labels)
    loss.backward()

    # update parameters
    optimizer.step()
    print(f"model now has weights\n{model.weights.data}\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    data = torch.randn(10, 10)
    labels = torch.randn(10, 10)
    weights = torch.randn(10, 10)
    start_time = time.time()
    process(data, labels, weights)
    print(f"total time: {time.time() - start_time:.3f} seconds")
