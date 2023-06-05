#!/bin/bash

declare -a files=(
    "plot_data.py"
    "plot_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
