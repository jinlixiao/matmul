import argparse
import torch
import torch.cuda as cuda
import torch.multiprocessing as mp
import models
import utils

NNODES = 1                 # Number of nodes
RUN_WITH_CPU = False       # Run with CPU instead of GPU
B, L, H = 24, 1024, 2560   # Batch size, sequence length, hidden size
EXCLUDE_ITERATIONS = 3     # Number of iterations to exclude from statistics

parser = argparse.ArgumentParser(description='Tiled Matrix Multiplication')
parser.add_argument('--num_tiles', type=int, default=1, help='Number of tiles to split the input into')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations to run')
parser.add_argument('--num_devices', type=int, default=torch.cuda.device_count(), help='Number of devices to use')
parser.add_argument('--model_name', type=str, default='OverlapTiledAllreduceMLP', help='Name of the model')
args = parser.parse_args()

MODEL_PARALLEL_SIZE = args.num_devices    # Number of GPUs per node
USE_TILES = args.model_name in ['OverlapTiledAllreduceMLP', 'OverlapTiledAllgatherMLP', 'OverlapTiledReduceScatterMLP']

def run(rank, world_size):
    utils.parallel_init(rank, world_size, run_with_cpu=RUN_WITH_CPU)
    model_parallel_group = utils.get_model_parallel_group()
    assert model_parallel_group is not None

    # Create the MLP model
    model = models.get_model_class(args.model_name)
    if RUN_WITH_CPU:
        mlp = model(H, 4 * H // MODEL_PARALLEL_SIZE)
        input_ = torch.randn(B, L, H)
    else:
        mlp = model(H, 4 * H // MODEL_PARALLEL_SIZE).cuda(rank)
        input_ = torch.randn(B, L, H).cuda(rank)

    # Run the model
    running_time = []
    with torch.no_grad():
        for i in range(args.num_iterations + EXCLUDE_ITERATIONS):
            utils.print_rank_0(f"********** Iteration {i} **********")
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)

            start_event.record()  # Start timing
            if USE_TILES:
                input_ = mlp(input_, num_tiles=args.num_tiles)
            else:
                input_ = mlp(input_)
            end_event.record()  # End timing
            cuda.synchronize()  # Wait for the events to be recorded
            
            duration = start_event.elapsed_time(end_event)
            if i >= EXCLUDE_ITERATIONS:
                running_time.append(duration)
            utils.print_rank_0(f"Rank {rank}: Time for tiled matrix multiplication: {duration:.04f} milliseconds")

    # Print statistics
    utils.print_rank_0(f"\n********** Statistics **********")
    utils.print_rank_0(f"Average time for tiled matrix multiplication: {sum(running_time) / len(running_time):.04f} milliseconds")
    utils.print_rank_0(f"Median time for tiled matrix multiplication: {sorted(running_time)[len(running_time) // 2]:.04f} milliseconds")

if __name__ == "__main__":
    if USE_TILES:
        print(f"Running {args.model_name} on {MODEL_PARALLEL_SIZE} GPUs per node, tile size {args.num_tiles}")
    else:
        print(f"Running {args.model_name} on {MODEL_PARALLEL_SIZE} GPUs per node")
    world_size = NNODES * MODEL_PARALLEL_SIZE
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
