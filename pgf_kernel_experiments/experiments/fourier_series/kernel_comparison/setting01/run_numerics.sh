#!/bin/bash

declare -a files=(
    "simulate_data.sh"
    "train_set_run_numerics.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   ./$file
done
