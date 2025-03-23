#!/bin/bash

# Loop for data_no_thrust_i
for i in {1..3}
do
    python3.8 plot.py data_no_thrust_$i
done

# Loop for data_thrust_i
for i in {1..3}
do
    python3.8 plot.py data_thrust_$i
done
