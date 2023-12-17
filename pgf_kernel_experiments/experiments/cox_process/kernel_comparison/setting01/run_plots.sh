#!/bin/bash

declare -a files=(
    "plot_projections.py"
    "plot_losses.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
