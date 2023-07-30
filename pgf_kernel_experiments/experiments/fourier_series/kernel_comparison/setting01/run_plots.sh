#!/bin/bash

declare -a files=(
    "train_set_run_plots.sh"
    "train_subset4_run_plots.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   ./$file
done
