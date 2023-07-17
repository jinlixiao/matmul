import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layers import LinearLayer

"""
Single Process 2-layer MLP
"""

SEED = 0
INPUT_SIZE = 4
BATCH_SIZE = 4
OUTPUT_SIZE = 4
HIDDEN_SIZE = 4

class Model(nn.Module):
    def __init__(self, weights1, weights2):
        super(Model, self).__init__()
        self.layer1 = LinearLayer(weights1)
        self.layer2 = LinearLayer(weights2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def process(data, labels, weights1, weights2):

    # model setup
    model = Model(weights1, weights2)
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
    print(f"model now has weights1\n{model.layer1.linear.weight.data}\n")
    print(f"model now has weights2\n{model.layer2.linear.weight.data}\n")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    data = torch.randn(BATCH_SIZE, INPUT_SIZE)
    labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)
    weights1 = torch.randn(INPUT_SIZE, HIDDEN_SIZE)
    weights2 = torch.randn(HIDDEN_SIZE, OUTPUT_SIZE)
    start_time = time.time()
    process(data, labels, weights1, weights2)
    print(f"total time: {time.time() - start_time:.3f} seconds")
