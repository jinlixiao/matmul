import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

EXCLUDE_ITERATIONS = 5  # Number of initial iterations to exclude

EXTRA_DESCRIPTION = "_2gpu_allgather"

def extract_times(output):
    """Extracts time values from the command output."""
    layer_times = []
    compute_times = []

    for line in output.split('\n'):
        match = re.search(r"Time for tiled matrix multiplication: ([\d.]+) milliseconds", line)
        if match:
            layer_times.append(float(match.group(1)))

        match = re.search(r"Total time for single_forward: ([\d.]+) milliseconds", line)
        if match:
            compute_times.append(float(match.group(1)))

    return layer_times, compute_times

layer_times_for_tile_sizes = {}
compute_times_for_tile_sizes = {}

for tile_size in [1]:
    file_path = f"output/time_{tile_size}{EXTRA_DESCRIPTION}.txt"
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Reading from file for tile size {tile_size}")
        with open(file_path, "r") as file:
            stdout = file.read()
    else:
        print(f"Running for tile size {tile_size}")
        # Run the command and capture its output
        command = f"python tiled_mlp_allgather.py --num_tiles {tile_size} --num_iter 100"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, stderr = process.communicate()

        # Save the output to a file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            file.write(stdout)
            file.write(stderr)

    # Extract times from the output or the file content
    layer_times, compute_times = extract_times(stdout)
    layer_times_for_tile_sizes[tile_size] = layer_times
    compute_times_for_tile_sizes[tile_size] = compute_times


# Box plot
tile_sizes = list(layer_times_for_tile_sizes.keys())
combined_times = []
for size in tile_sizes:
    # Exclude the first EXCLUDE_ITERATIONS
    combined_times.append(layer_times_for_tile_sizes[size][EXCLUDE_ITERATIONS:])
    combined_times.append(compute_times_for_tile_sizes[size][EXCLUDE_ITERATIONS:])
labels = []
for size in tile_sizes:
    labels.extend([f'Tile Size {size} Layer Times', f'Tile Size {size} Compute Times'])
plt.figure(figsize=(10, 6))
plt.boxplot(combined_times, labels=labels, showfliers=False)
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.title('Box-and-Whisker Plot for Layer Times and Compute Times')
plt.xlabel('Tile Size and Time Type')
plt.ylabel('Time (milliseconds)')
plt.grid(True)
plt.tight_layout()  # Adjust layout for better fit
plt.savefig(f'images/timing_analysis_combined_boxplot_iter100{EXTRA_DESCRIPTION}.png')
plt.show()

# Grouped bar chart
layer_means = [np.mean(layer_times_for_tile_sizes[size][EXCLUDE_ITERATIONS:]) for size in tile_sizes]
compute_means = [np.mean(compute_times_for_tile_sizes[size][EXCLUDE_ITERATIONS:]) for size in tile_sizes]
x = np.arange(len(tile_sizes))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, layer_means, width, label='Layer Times')
rects2 = ax.bar(x + width/2, compute_means, width, label='Compute Times')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Tile Size')
ax.set_ylabel('Time (milliseconds)')
ax.set_title('Layer Times vs Compute Times by Tile Size')
ax.set_xticks(x)
ax.set_xticklabels(tile_sizes)
ax.legend()
plt.savefig(f'images/timing_analysis_combined_bar_chart_iter100{EXTRA_DESCRIPTION}.png')
plt.show()

# Pretty-printing the times
print("Layer Times and Compute Times (in milliseconds):")
for i, size in enumerate(tile_sizes):
    print(f"Tile Size {size}: Layer Time = {layer_means[i]:.2f} ms, Compute Time = {compute_means[i]:.2f} ms")