#!/bin/bash

declare -a files=(
    "train_set_plot_data.py"
    "train_set_plot_predictions.py"
    "train_set_plot_losses.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
