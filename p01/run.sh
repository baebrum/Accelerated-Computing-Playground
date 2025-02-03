#!/bin/bash

# Check if exactly one argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <power_of_two>"
  exit 1
fi

# Get the argument (should be a power of two or 1)
arg=$1

# Check if the argument is a valid power of two or 1
if ! [[ "$arg" =~ ^[0-9]+$ ]] || (( arg <= 0 )) || (( arg != 1 && arg & (arg - 1) != 0 )); then
  echo "Error: Argument must be a power of two or 1."
  exit 1
fi

rm -rf _output
mkdir -p _output

# Output CSV headers (if file doesn't exist)
output_file="_output/performance_data.csv"
if [ ! -f "$output_file" ]; then
  echo "Time_Seconds,Num_Threads" > "$output_file"
fi

# Loop over powers of two, from 1 to the given argument
for ((i = 1; i <= arg; i *= 2)); do
  echo "Running with argument $i..."

  # Run the program and capture the output
  program_output=$(./_build/p01 $i)

  # Extract time and number of threads from the output
  time=$(echo "$program_output" | grep -oP 'Order \d+ multiplication in \K[0-9.]+')
  threads=$(echo "$program_output" | grep -oP 'Setting OpenMP to use \K\d+')

  # If we couldn't extract the time or threads, skip this iteration
  if [ -z "$time" ] || [ -z "$threads" ]; then
    echo "Error: Failed to extract time or threads from program output."
    continue
  fi

  # Append the data to the CSV file
  echo "$time,$threads" >> "$output_file"
done
