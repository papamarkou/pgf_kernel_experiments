#!/bin/bash

declare -a files=(
    "simulate_data.py"
    "train_set_run_numerics.sh"
    "train_subset_run_numerics.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
