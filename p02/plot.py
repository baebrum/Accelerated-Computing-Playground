import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.ticker import FuncFormatter

# Parse input file and extract data into pandas DataFrame
def parse_input_file(input_file):
    data = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            first_line = lines[i].strip()
            second_line = lines[i + 1].strip()

            N = int(first_line.split('=')[1].split('and')[0].strip())
            threads = int(first_line.split('=')[2].strip().split()[0])
            workload = float(second_line.split(':')[1].strip().split()[0])

            data.append([N, threads, workload])

    return pd.DataFrame(data, columns=['N', 'num_threads', 'workload_seconds'])

# Save data to CSV using pandas
def save_to_csv(df, output_csv_file):
    df.to_csv(output_csv_file, index=False)

# Format x-axis ticks for log scale
def format_log_ticks(value, tick_pos):
    return f"$2^{{{int(np.log2(value))}}}$"

# Plot log scale graph using pandas plotting functionality
def plot_log_scale(df, output_file, exclude_threads=[]):
    plt.figure(figsize=(10, 6))

    cmap = plt.colormaps["tab10"]
    colors = [cmap(i / len(df['num_threads'].unique())) for i in range(len(df['num_threads'].unique()))]

    for idx, threads in enumerate(df['num_threads'].unique()):
        if threads in exclude_threads: continue
        subset = df[df['num_threads'] == threads].sort_values(by='N')

        line_style, marker_style, color = ('-', 'o', 'black') if threads == 1 else ('--', 'x', colors[idx])
        plt.plot(subset['N'], subset['workload_seconds'], label=f'{threads} Threads', color=color, linestyle=line_style, marker=marker_style)

    max_N = df['N'].max()
    ticks = [2**i for i in range(int(np.log2(max_N)) + 2)]
    plt.xscale('log', base=2)
    plt.xticks(ticks)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_log_ticks))
    plt.xlabel('Node Count (N)')
    plt.ylabel('Workload Time (t) in seconds')
    plt.title(f'Workload Time vs Node Count (N) for Different Thread Counts{" Excluding " + ", ".join(map(str, exclude_threads)) if exclude_threads else ""}')
    plt.legend(title='Thread Count', loc='best')
    plt.grid(True)
    plt.savefig(output_file)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process workload data from a text file.")
    parser.add_argument('input_file', type=str, help="Path to the input text file")
    args = parser.parse_args()

    df = parse_input_file(args.input_file)
    save_to_csv(df, 'output_data.csv')

    for exclude_threads, output_file in [
        ([], 'workload_plot_log_with_1thread.png'),
        ([1], 'workload_plot_log_without_1thread.png'),
        ([1, 2, 4], 'workload_plot_log_without_1_2_4threads.png'),
        ([1, 2, 4, 8], 'workload_plot_log_without_1_2_4_8threads.png'),
    ]:
        plot_log_scale(df, output_file, exclude_threads)
        plt.clf()

    print(f"Data saved to output_data.csv and plots saved to 'workload_plot_log_with_1thread.png', "
          "'workload_plot_log_without_1thread.png', and 'workload_plot_log_without_1_2_4threads.png'.")

if __name__ == "__main__":
    main()
