import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

EXCLUDE_ITERATIONS = 5  # Number of initial iterations to exclude
SAVE_FIG = True         # Save the figure to a file
OVERWRITE = True        # Overwrite existing files

EXTRA_DESCRIPTION = "2gpu_nonoverlap"

def extract_times(output):
    """Extracts time values from the command output."""
    layer_times = []
    compute_times = []
    reduce_scatter_times = []
    all_gather_times = []

    for line in output.split('\n'):
        match = re.search(r"Time for nontiled matrix multiplication: ([\d.]+) milliseconds", line)
        if match:
            layer_times.append(float(match.group(1)))

        match = re.search(r"Total time for matrix multiplications: ([\d.]+) milliseconds", line)
        if match:
            compute_times.append(float(match.group(1)))
        
        match = re.search(r"Total time for reduce_scatter: ([\d.]+) milliseconds", line)
        if match:
            reduce_scatter_times.append(float(match.group(1)))
        
        match = re.search(r"Total time for all_gather: ([\d.]+) milliseconds", line)
        if match:
            all_gather_times.append(float(match.group(1)))

    return layer_times, compute_times, reduce_scatter_times, all_gather_times

# Run the experiment and capture the output
file_path = f"output/nontiled/time_{EXTRA_DESCRIPTION}.txt"

# Check if the file already exists
if not OVERWRITE and os.path.exists(file_path):
    print(f"Reading from file...")
    with open(file_path, "r") as file:
        stdout = file.read()
else:
    print(f"Running for {EXTRA_DESCRIPTION}")
    # Run the command and capture its output
    command = f"python nontiled_mlp.py --num_iter 100"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    stdout, stderr = process.communicate()

    # Save the output to a file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(stdout)
        file.write(stderr)

# Extract times from the output or the file content
layer_times, compute_times, reduce_scatter_times, all_gather_times = extract_times(stdout)

# Exclude the first EXCLUDE_ITERATIONS
layer_times = layer_times[EXCLUDE_ITERATIONS:]
compute_times = compute_times[EXCLUDE_ITERATIONS:]
reduce_scatter_times = reduce_scatter_times[EXCLUDE_ITERATIONS:]
all_gather_times = all_gather_times[EXCLUDE_ITERATIONS:]

layer_mean = np.mean(layer_times)
compute_mean = np.mean(compute_times)
reduce_scatter_mean = np.mean(reduce_scatter_times)
all_gather_mean = np.mean(all_gather_times)

# Pretty-printing the times
print("Layer Times and Compute Times (in milliseconds):")
print(f"Layer Time = {layer_mean:.2f} ms, Compute Time = {compute_mean:.2f} ms, Reduce Scatter Time = {reduce_scatter_mean:.2f} ms, All Gather Time = {all_gather_mean:.2f} ms")