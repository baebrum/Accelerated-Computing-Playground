import matplotlib.pyplot as plt
import sys
import numpy as np
import os

def parse_file(file_name):
    cpu_times = []
    gpu_times = []
    exponents = []

    with open(file_name, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            # Check for the line starting with "Running reduction"
            if lines[i].startswith("Running reduction"):
                # calculate N from the first value in the CPU line
                cpu_data = lines[i+1].split(', ')
                first_value = float(cpu_data[0])  # First value in CPU line
                N_exponent = int(np.log2(first_value))  # log2 to get the exponent N

                # Extracting time for CPU
                cpu_time = float(cpu_data[1])

                # Extracting time for GPU
                gpu_data = lines[i+2].split(', ')
                gpu_time = float(gpu_data[1])

                # Store the results
                exponents.append(N_exponent)
                cpu_times.append(cpu_time)
                gpu_times.append(gpu_time)

                # Increment by 4 lines: 1 for header, 1 for CPU data, 1 for GPU data, 1 for separator
                i += 4
            else:
                # Skip lines that are not in the correct format
                i += 1

    return exponents, cpu_times, gpu_times

def plot_data(exponents, cpu_times, gpu_times, save_path):
    # Create the main plot for CPU and GPU times
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot CPU times
    ax1.plot(exponents, cpu_times[:len(exponents)], label='CPU', marker='o', linestyle='-', color='b')

    # Plot GPU times
    ax1.plot(exponents, gpu_times[:len(exponents)], label='GPU', marker='o', linestyle='-', color='r')

    # Adding labels and title for the first axis
    ax1.set_xlabel("N")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("N vs Time for CPU and GPU (Summation from 1 to N)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Set the x-axis ticks to be powers of 2
    ticks = np.arange(min(exponents), max(exponents) + 1, 1)  # Create ticks from exponent range
    tick_labels = [f'$2^{{{i}}}$' for i in ticks]  # Power of 2 labels (formatted)

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels)  # Set the formatted tick labels

    # Create the second y-axis for speedup
    ax2 = ax1.twinx()

    # Calculate speedup
    speedup = [cpu / gpu if gpu != 0 else 0 for cpu, gpu in zip(cpu_times, gpu_times)]

    # Plot speedup on the second y-axis
    ax2.plot(exponents, speedup, label='Speedup', marker='x', linestyle='--', color='g')

    # Highlight the breakeven point
    breakeven_index = next((i for i, s in enumerate(speedup) if s > 1), None)
    if breakeven_index is not None:
        ax2.scatter(exponents[breakeven_index], speedup[breakeven_index], color='purple', zorder=5, label="Breakeven Point")
        ax2.axvline(x=exponents[breakeven_index], color='purple', linestyle=':', label="Breakeven Line")

    # Adding the second y-axis label with subscripts
    ax2.set_ylabel(r"Speedup ($t_{sequential}$ / $t_{parallel}$)", color='g')  # Using subscripts
    ax2.tick_params(axis='y', labelcolor='g')

    # Save the plot as a .png file
    plt.legend(loc="upper right")
    plt.savefig(save_path, format='png')  # Save the plot as PNG
    plt.close()  # Close the plot to avoid showing it

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input file as an argument.")
        sys.exit(1)

    file_name = sys.argv[1]
    exponents, cpu_times, gpu_times = parse_file(file_name)

    # Get the base name of the file without extension and append .png
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    save_path = f"{base_name}.png"  # Output file path

    plot_data(exponents, cpu_times, gpu_times, save_path)
    print(f"Plot saved as {save_path}")
