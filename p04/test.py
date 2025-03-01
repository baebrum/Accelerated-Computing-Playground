import subprocess
import re

def run_jacobi(threads):
    """Runs the Jacobi application with the given number of threads and returns the execution time."""
    cmd = f"./jacobi_cpu -n 100 -i 10000000 -t {threads}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    match = re.search(r'Time taken for Jacobi iterations: ([0-9\.]+) seconds', result.stdout)
    return float(match.group(1)) if match else None

def compute_speedup():
    """Runs Jacobi for 1 thread, then computes speedup for increasing thread counts."""
    base_time = run_jacobi(1)
    if base_time is None:
        print("Error: Could not determine baseline execution time.")
        return
    print(f"Threads: 1, Execution Time: {base_time:.6f}, Speedup: 1.00")

    for exp in range(1, 8):  # 2^1 to 2^7
        threads = 2 ** exp
        exec_time = run_jacobi(threads)
        if exec_time is None:
            print(f"Error: Could not determine execution time for {threads} threads.")
            break

        speedup = base_time / exec_time
        print(f"Threads: {threads}, Execution Time: {exec_time:.6f}, Speedup: {speedup:.2f}")

if __name__ == "__main__":
    compute_speedup()
