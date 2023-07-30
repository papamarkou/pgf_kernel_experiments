#!/bin/bash

declare -a files=(
    "simulate_data.sh"
    "train_set_run_numerics.sh"
    "train_subset4_run_numerics.sh"
    "train_subset3_run_numerics.sh"
    "train_subset2_run_numerics.sh"
    "train_subset1_run_numerics.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   ./$file
done
