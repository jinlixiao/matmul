import sys
import unittest
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layer_no_partition import LinearLayer
from layer_data_partition import DataParallelLinearLayer
from layer_model_partition import ModelParallelLinearLayer
from partition import partition_tensor

class TestLayers(unittest.TestCase):

    def test_data_parallel_model(self):
        self._test_model(_run_data_parallel_model)
    
    def test_model_parallel_model(self):
        self._test_model(_run_model_parallel_model)

    def _test_model(self, proc_func, world_size=2, input_size=10, output_size=10, batch_size=10):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "64758"
        torch.manual_seed(0)
        data = torch.randn(batch_size, input_size)
        labels = torch.randn(batch_size, output_size)
        weights = torch.randn(input_size, output_size)

        # expected result
        weights_clone = weights.detach().clone()
        data_clone = data.detach().clone()
        label_clone = labels.detach().clone()
        expected_weights = _run_linear_model_one_pass(data_clone, label_clone, weights_clone)

        mp.spawn(proc_func, args=(world_size, data, labels, weights, expected_weights), nprocs=world_size, join=True)

def _run_linear_model_one_pass(data, label, weights):
    """
    Run one pass of linear model, return the updated weights
    """
    model = LinearLayer(weights)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    outputs = model(data)
    loss = loss_fn(outputs, label)
    loss.backward()
    optimizer.step()
    return model.linear.weight.data
    

def _run_data_parallel_model(rank, world_size, data, label, weights, expected_weights):
    
    # actual result
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # setup model
    data = partition_tensor(data, world_size, rank, dim=0)
    label = partition_tensor(label, world_size, rank, dim=0)
    model = DataParallelLinearLayer(weights, group)
    model = model.to(rank)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # forward pass
    data = data.to(rank)
    outputs = model(data.to(rank))
    
    # backward pass
    label = label.to(rank)
    loss = loss_fn(outputs, label)
    loss.backward()

    # update parameters
    optimizer.step()
    actual_weights = model.linear.weight.data.to('cpu')

    assert torch.equal(actual_weights, expected_weights), "Model parallel model does not match expected result"

def _run_model_parallel_model(rank, world_size, data, label, weights, expected_weights):

    # actual result
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.new_group([0, 1])

    # model setup
    weights = partition_tensor(weights, world_size, rank, dim=1)
    model = ModelParallelLinearLayer(weights, group)
    model = model.to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # forward pass
    outputs = model(data.to(rank))

    # backward pass
    label = label.to(rank)
    loss = loss_fn(outputs, label)
    loss.backward()

    # update parameters
    optimizer.step()
    actual_weights = model.linear.weight.data.to('cpu')

    # compare result
    expected_weights = partition_tensor(expected_weights, world_size, rank, dim=0)
    assert torch.equal(actual_weights, expected_weights), "Model parallel model does not match expected result"


if __name__ == '__main__':
    unittest.main()
