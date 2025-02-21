#!/bin/bash

# set -x
# Path to the compiled program
PROGRAM="./_build/src/main"

# Create a timestamp for the output file name
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="results_${timestamp}.csv"

ITER=0
STARTING_NUM_THREADS=1
STARTING_NUM_N=1 # (this should be read as 2^STARTING_NUM_N)
TOTAL_INTERVALS=21 # (this should be read as 2^TOTAL_INTERVALS)
TOTALTHREADS=8 # (this should be read as 2^TOTALTHREADS)
# Iterate over N (2^0 to 2^18)
echo "N,num_threads,wall_clock_time" > "$output_file"
for ((N=2**STARTING_NUM_N; N<=2**TOTAL_INTERVALS; N*=2)); do
    # Iterate over number of threads (1, 2^0, 2^1, ..., 2^8)
    # echo "$N is N; $ITER is iter"
    for ((threads=STARTING_NUM_THREADS; threads<=2**TOTALTHREADS; threads*=2)); do
        # Run the program with the given N and number of threads, and append output to the file
        # $PROGRAM -n $N -t $threads >> "$output_file"; echo " " >> "$output_file" # output user readable performance to file
        $PROGRAM -n $N -t $threads -o csv >> "$output_file" # output csv friendly to file
        # $PROGRAM -n $N -t $threads # output to standard output (not csv friendly)
    done
    ((ITER+=1))
done
