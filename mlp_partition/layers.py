import torch
import torch.distributed as dist
import torch.nn as nn


###############
# No Parallel
###############


class LinearLayer(nn.Module):
    def __init__(self, weights):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(*weights.t().shape, bias=False)
        self.linear.weight.data = weights.t().clone()

    def forward(self, input):
        return self.linear(input)


###############
# Data Parallel
###############


class DataParallelLinearLayer(nn.Module):
    """
    Data parallel linear model
    """

    def __init__(self, weights, group):
        super(DataParallelLinearLayer, self).__init__()
        self.linear = nn.Linear(*weights.t().shape, bias=False)
        self.linear.weight.data = weights.t().clone()
        self.linear.weight.register_hook(self._all_reduce_hook)
        self.group = group

    def forward(self, input):
        output = self.linear(input)
        return output

    def _all_reduce_hook(self, grad_weight):
        dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=self.group)
        return grad_weight / dist.get_world_size()


################
# Model Parallel
################


class ModelParallelLinearLayer(nn.Module):
    def __init__(self, weights, group):
        super(ModelParallelLinearLayer, self).__init__()
        self.linear = nn.Linear(*weights.t().shape, bias=False)
        self.linear.weight.data = weights.t().clone()
        self.group = group

    def forward(self, input):
        output_local = self.linear(input)
        output_list = [
            torch.empty_like(output_local) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output_list, output_local, group=self.group)
        output_list[dist.get_rank(self.group)] = output_local
        output = torch.cat(output_list, dim=1).contiguous()
        return output
