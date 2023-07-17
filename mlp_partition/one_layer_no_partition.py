import time
import torch
import torch.nn as nn
import torch.optim as optim

from models import OneLayerNoPartitionModel

"""
Single Process Model: one CPU
"""

SEED = 0
INPUT_SIZE = 4
BATCH_SIZE = 4
OUTPUT_SIZE = 4

def process(data, labels, weights):

    # model setup
    model = OneLayerNoPartitionModel(weights)
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
    print(f"model now has weights\n{model.layer.linear.weight.data}\n")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    data = torch.randn(BATCH_SIZE, INPUT_SIZE)
    labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)
    weights = torch.randn(INPUT_SIZE, OUTPUT_SIZE)
    start_time = time.time()
    process(data, labels, weights)
    print(f"total time: {time.time() - start_time:.3f} seconds")
