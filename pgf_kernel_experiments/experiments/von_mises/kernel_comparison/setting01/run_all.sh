#!/bin/bash

declare -a files=(
    "simulate_data.py"
    "train_gps.py"
    "make_predictions.py"
    "plot_data.py"
    "plot_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
