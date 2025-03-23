#!/bin/bash

# Loop over the specified number of tests
for test in {1..3}
do
  # Define output files and clear them at the start of each test
  no_thrust_file="data_no_thrust_${test}"
  thrust_file="data_thrust_${test}"

  > "$no_thrust_file"  # Clears the file (or creates it if it doesn’t exist)
  > "$thrust_file"      # Clears the file (or creates it if it doesn’t exist)

  for power in {0..25}
  do
      N=$((2 ** power))
      echo "Running reduction with 2^{N = $power}" | tee -a "$no_thrust_file"
      ./mainCUDA $N >> "$no_thrust_file"
      echo "--------------------------------------------" >> "$no_thrust_file"
  done

  for power in {0..25}
  do
      N=$((2 ** power))
      echo "Running reduction with 2^{N = $power}" | tee -a "$thrust_file"
      ./mainCUDA_thrust $N >> "$thrust_file"
      echo "--------------------------------------------" >> "$thrust_file"
  done
done
