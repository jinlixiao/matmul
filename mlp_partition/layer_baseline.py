import time
import torch
import torch.nn as nn
import torch.optim as optim

from layers import LinearModel

"""
Single Process Model: one CPU
"""

def process(data, labels, weights):

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

    torch.set_printoptions(sci_mode=False, precision=4)
    print(f"model now has weights\n{model.linear.weight.data}\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    data = torch.randn(4, 4)
    labels = torch.randn(4, 4)
    weights = torch.randn(4, 4)
    start_time = time.time()
    process(data, labels, weights)
    print(f"total time: {time.time() - start_time:.3f} seconds")
