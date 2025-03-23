import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def parse_input_file(file_path):
    p_values = set()
    data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Matrix dimensions"):
                # Extract m, p, n
                m_p_n = lines[i].strip().split(", ")
                m = int(m_p_n[0].split("=")[1].strip())
                p = int(m_p_n[1].split("=")[1].strip())
                n = int(m_p_n[2].split("=")[1].strip())

                # Check if p is a power of 2
                if (p & (p - 1)) == 0 and p > 0:  # If p is a power of 2
                    # Add p to the set of unique p values
                    p_values.add(p)

                # Skip two lines and get the times
                basic, tiled = map(float, lines[i + 2].strip().split(","))

                if p not in data:
                    data[p] = {'m_values': [], 'basic_times': [], 'tiled_times': []}

                data[p]['m_values'].append(m)  # Assuming m = n
                data[p]['basic_times'].append(basic)
                data[p]['tiled_times'].append(tiled)

                i += 2
            else:
                i += 1

    return p_values, data

def plot_data(m_values, basic_times, tiled_times, p_value):
    # Calculate the power of 2 for p (i.e., the exponent)
    power_of_2 = int(math.log2(p_value))

    plt.figure(figsize=(8, 6))

    # Plot Basic Matrix Multiplication
    plt.plot(m_values, basic_times, color='blue',
             label=f'Basic ($p = 2^{{{power_of_2}}}$)', linestyle='-', marker='o')

    # Plot Tiled Matrix Multiplication
    plt.plot(m_values, tiled_times, color='red',
             label=f'Tiled ($p = 2^{{{power_of_2}}}$)', linestyle='-', marker='x')

    plt.xscale('log', base=2)  # Use a logarithmic scale for the x-axis with base 2
    plt.xlabel('Matrix Dimension $m = n$ (powers of 2)', fontsize=12)
    plt.ylabel('Kernel Elapsed Time (ms)', fontsize=12)
    plt.title(f'Matrix Multiplication Kernel Elapsed Time vs. $m = n$ ($p = 2^{{{power_of_2}}}$)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Ensure all x-axis ticks are shown for powers of 2
    min_m, max_m = min(m_values), max(m_values)
    x_ticks = [2**i for i in range(int(math.log2(min_m)), int(math.log2(max_m)) + 1)]
    plt.xticks(x_ticks, [f'$2^{{{i}}}$' for i in range(int(math.log2(min_m)), int(math.log2(max_m)) + 1)])


    # Save the plot to a file based on power of 2 (log2(p))
    output_file = f"plot_power_2_{power_of_2}.png"
    plt.tight_layout()
    plt.savefig(output_file)  # Save the plot as an image file (e.g., PNG or PDF)
    print(f"Plot saved as {output_file}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot Matrix Multiplication Elapsed Time')
    parser.add_argument('file', type=str, help='Path to the input data file')
    args = parser.parse_args()

    # Parse the input file and collect unique p values and data
    p_values, data = parse_input_file(args.file)

    # Plot and save the data for each p value
    for p_value in sorted(p_values):
        m_values = np.array(data[p_value]['m_values'])
        basic_times = np.array(data[p_value]['basic_times'])
        tiled_times = np.array(data[p_value]['tiled_times'])

        # Plot and save the plot for the current p_value
        plot_data(m_values, basic_times, tiled_times, p_value)

if __name__ == "__main__":
    main()
