import argparse
from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_wall_clock_time(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    # Loop through unique numbers of threads
    for num_threads in df['num_threads'].unique():
        # Filter the DataFrame for the current number of threads
        thread_data = df[df['num_threads'] == num_threads]

        # Plot N vs. wall clock time for the current number of threads
        plt.plot(thread_data['N'], thread_data['wall_clock_time'], label=f'{num_threads} threads', marker='o')

    # Set the x-axis to be logarithmic (base 2)
    plt.xscale('log', base=2)

    # Adding labels and title
    plt.xlabel('N (Number of Intervals)')
    plt.ylabel('Wall Clock Time (seconds)')
    plt.title('Wall Clock Time vs. N for Different Numbers of Threads')

    # Set the x-axis ticks to be powers of 2
    min_x = int(np.log2(df['N'].min()))  # Smallest power of 2 in the data
    max_x = int(np.log2(df['N'].max()))  # Largest power of 2 in the data
    ticks = [2**i for i in range(min_x, max_x + 1)]  # List of powers of 2
    tick_labels = [f'$2^{{{i}}}$' for i in range(min_x, max_x + 1)]

    plt.xticks(ticks, labels=tick_labels)  # Set the formatted tick labels

    # Add a legend and grid
    plt.legend(title='Threads', loc='upper left')
    plt.grid(True)

    # Save the first plot to a file
    plt.savefig(output_file)

    # Close the plot to avoid display (since we're saving it)
    plt.close()

def plot_runtime_vs_threads(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Find the row corresponding to 2^20 intervals (N=2^20)
    target_N = 2**20
    df_at_target_N = df[df['N'] == target_N]

    # Find the minimum wall clock time and the corresponding number of threads
    min_time_row = df_at_target_N.loc[df_at_target_N['wall_clock_time'].idxmin()]
    min_time = min_time_row['wall_clock_time']
    min_threads = min_time_row['num_threads']

    # Plot runtime vs. thread count for N = 2^20
    plt.figure(figsize=(10, 6))
    plt.plot(df_at_target_N['num_threads'], df_at_target_N['wall_clock_time'], marker='o', label='Wall Clock Time')

    # Plot red dot for the minimum time
    plt.plot(min_threads, min_time, 'ro', label=f'Min Time: {min_time:.4f}s at {min_threads} threads')

    # Set the x-axis to be logarithmic (base 2)
    plt.xscale('log', base=2)

    # Adding labels and title
    plt.xlabel('Number of Threads')
    plt.ylabel('Wall Clock Time (seconds)')
    plt.title(f'Runtime vs. Thread Count for N = {target_N} = $2^{{{20}}}$')

    # Set the x-axis ticks to be powers of 2
    min_x = int(np.log2(df_at_target_N['num_threads'].min()))  # Smallest power of 2 in the data
    max_x = int(np.log2(df_at_target_N['num_threads'].max()))  # Largest power of 2 in the data
    ticks = [2**i for i in range(min_x, max_x + 1)]  # List of powers of 2
    tick_labels = [f'$2^{{{i}}}$' for i in range(min_x, max_x + 1)]

    plt.xticks(ticks, labels=tick_labels)  # Set the formatted tick labels

    # Add a legend and grid
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the runtime vs. thread count plot to a file
    plt.savefig(output_file)

    # Close the plot
    plt.close()

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot N vs wall clock time for different numbers of threads.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the results.')
    parser.add_argument('output_file1', type=str, help='Path to save the first output plot (wall clock time vs. N).')
    parser.add_argument('output_file2', type=str, help='Path to save the second output plot (runtime vs. thread count for N=2^20).')

    # Display the usage if no arguments are provided
    if len(argv) == 1:
        print("Usage: script.py <csv_file> <output_file1> <output_file2>")
        print("Please provide the input CSV file and both output file paths.")
        exit()

    args = parser.parse_args()

    # Print the received arguments
    print(f"Arguments received: CSV file: {args.csv_file}, Output file 1: {args.output_file1}, Output file 2: {args.output_file2}")

    # Call the plot function for wall clock time vs. N
    plot_wall_clock_time(args.csv_file, args.output_file1)

    # Call the plot function for runtime vs. thread count at N=2^20
    plot_runtime_vs_threads(args.csv_file, args.output_file2)
