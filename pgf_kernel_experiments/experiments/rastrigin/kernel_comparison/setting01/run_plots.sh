#!/bin/bash

declare -a files=(
    "plot_data.py"
    "plot_predictions.py"
    "plot_losses.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
