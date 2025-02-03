import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file using pandas
df = pd.read_csv('_output/performance_data.csv')

# Find the row with the minimum time
min_row = df.loc[df['Time_Seconds'].idxmin()]
min_time = min_row['Time_Seconds']
min_threads = min_row['Num_Threads']

# Plot the performance
plt.figure(figsize=(10, 6))

# Plot the main data
plt.plot(df['Num_Threads'], df['Time_Seconds'], marker='o', label='Time (seconds)')

# Highlight the point with the minimum time
plt.scatter(min_threads, min_time, color='red', s=100, zorder=5,
            label=f'Min Time: {min_time:.4f}s at {min_threads} threads')

# Set axis labels
plt.xlabel('Hardware Threads')
plt.ylabel('Elapsed (wall clock time), s')
plt.title('Performance of Matrix Multiplication vs. Hardware Threads')

# Set X-axis scale to log
plt.xscale('log')

# Set X-axis limits from 2^0 to 2^9 (1 to 512)
plt.xlim(1, 512)

# Set Y-axis limits from 0 to 4 seconds
plt.ylim(0, 4)

x_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
plt.xticks(x_ticks, [f'$2^{i}$' for i in range(len(x_ticks))])

y_ticks = np.linspace(0, 4, 5)  # Tick marks from 0 to 8 seconds

plt.yticks(y_ticks)
plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
# Plot the legend
plt.legend()
# Show the plot
plt.show()
