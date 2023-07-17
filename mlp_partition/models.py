import torch
import torch.distributed as dist
import torch.nn as nn

import layers
from partition import partition_tensor

###############
# One Layer
###############


class OneLayerNoPartitionModel(nn.Module):
    def __init__(self, weights):
        super(OneLayerNoPartitionModel, self).__init__()
        self.layer = layers.LinearLayer(weights)

    def forward(self, input):
        return self.layer(input)


class OneLayerDataPartitionModel(nn.Module):
    def __init__(self, weights, group):
        super(OneLayerDataPartitionModel, self).__init__()
        self.layer = layers.DataParallelLinearLayer(weights, group)

    def forward(self, input):
        return self.layer(input)


class OneLayerModelPartitionModel(nn.Module):
    def __init__(self, weights, group):
        super(OneLayerModelPartitionModel, self).__init__()
        weights = partition_tensor(
            weights, dist.get_world_size(), dist.get_rank(group), dim=1
        )
        self.layer = layers.ModelParallelLinearLayer(weights, group)

    def forward(self, input):
        return self.layer(input)


###############
# Two Layers
###############


class TwoLayerDataPartitionModel(nn.Module):
    def __init__(self, weights1, weights2, group):
        super(TwoLayerDataPartitionModel, self).__init__()
        self.layer1 = layers.DataParallelLinearLayer(weights1, group)
        self.layer2 = layers.DataParallelLinearLayer(weights2, group)

    def forward(self, input):
        output = self.layer1(input)
        output = torch.relu(output)
        output = self.layer2(output)
        return output
