import json
import matplotlib.pyplot as plt

# Load timing data
with open('timing_data.json', 'r') as f:
    timing_data = json.load(f)

# Prepare data for boxplot
data_to_plot = [times for times in timing_data.values()]
labels = [f'Tile Size {tile_size}' for tile_size in timing_data.keys()]

# Plotting
plt.figure()
plt.boxplot(data_to_plot, labels=labels, showfliers=False)  # Do not show outliers

plt.xlabel('Tile Size')
plt.ylabel('Time (seconds)')
plt.title('Box and Whisker Plot of Timing Analysis for Different Tile Sizes (Outliers Excluded)')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.grid(True)  # Optional: Adds a grid for easier reading
plt.savefig('timing_analysis_boxplot_no_outliers.png')
