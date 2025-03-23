#!/bin/bash

# Define the upper limit
LIMIT=15

# Output file
OUTPUT_FILE="output.log"
> "$OUTPUT_FILE"  # Clear the file before running

# Loop through values of x and y from 0 to LIMIT
for x in $(seq 0 $LIMIT); do
  for y in $(seq 0 $LIMIT); do
    ./main "$x" "$y" >> "$OUTPUT_FILE"
  done
done

wait
echo "All tasks completed. Output stored in $OUTPUT_FILE"
