#!/bin/bash

declare -a files=(
    "train_set_run_plots.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
