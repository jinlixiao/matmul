import subprocess
import json

tile_sizes = [1, 2, 4, 8, 16, 32]
repetitions = 100
timing_data = {}

for tile_size in tile_sizes:
    times = []
    for _ in range(repetitions):
        result = subprocess.run(['python', 'tiled_mlp.py', '--num_tiles', str(tile_size)], capture_output=True, text=True)
        time_taken = float(result.stdout.split()[-2])  # Extract the time from output
        times.append(time_taken)
        print(f"Tile Size: {tile_size}, Time Taken: {time_taken:.04f} seconds")
    timing_data[tile_size] = times

# Save the timing data
with open('timing_data.json', 'w') as f:
    json.dump(timing_data, f)

print(timing_data)
