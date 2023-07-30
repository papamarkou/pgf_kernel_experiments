#!/bin/bash

declare -a files=(
    "train_subset4_plot_data.py"
    "train_subset4_plot_predictions.py"
    "train_subset4_plot_losses.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
