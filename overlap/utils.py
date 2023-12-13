import argparse
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Parallel Settings

_MODEL_PARALLEL_GROUP = None

def parallel_init(rank, world_size, run_with_cpu=False):
    if not run_with_cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    if run_with_cpu:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = dist.new_group(range(world_size))

def get_model_parallel_group():
    global _MODEL_PARALLEL_GROUP
    return _MODEL_PARALLEL_GROUP


# Print settings
def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)