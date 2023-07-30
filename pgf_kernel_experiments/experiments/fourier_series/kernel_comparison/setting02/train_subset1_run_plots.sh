#!/bin/bash

declare -a files=(
    "train_subset1_plot_data.py"
    "train_subset1_plot_predictions.py"
    "train_subset1_plot_losses.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
