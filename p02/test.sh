#!/bin/bash

# Path to the compiled program
PROGRAM="./_build/src/main"

# Create a timestamp for the output file name
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="output_${timestamp}.txt"

ITER=0
STARTING_NUM_THREADS=4
TOTALNODES=18 # (this should be read as 2^TOTALNODES)
TOTALTHREADS=8 # (this should be read as 2^TOTALTHREADS)
# Iterate over N (2^0 to 2^18)
for ((N=2; N<=2**TOTALNODES; N*=2)); do
    # Iterate over number of threads (1, 2^0, 2^1, ..., 2^8)
    echo "$N is N; $ITER is iter"
    for ((threads=STARTING_NUM_THREADS; threads<=2**TOTALTHREADS; threads*=2)); do
        # Run the program with the given N and number of threads, and append output to the file
        $PROGRAM -n $N -t $threads >> "$output_file"
    done
    ((ITER+=1))
done
