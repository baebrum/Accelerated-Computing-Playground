#!/bin/bash

# Define the specific number of threads to run
threads_list=(1 8 16 32 64)

# Loop over the specified number of threads
for threads in "${threads_list[@]}"
do
    # Run the program in the background with the current number of threads using nohup
    # Redirecting stdout to a unique file for each run and running in the background
    nohup ./jacobi_cpu -n 1000 -i 5123000 -t $threads > ./jacobi_cpu_1000_5123000_${threads}_stdout 2>&1 &

    # Optional: Print a message indicating the program is running with the current number of threads
    echo "Running jacobi_cpu with $threads threads in the background..."
done

echo "All tasks are now running in the background."
