#!/bin/bash

declare -a files=(
    "simulate_data.py"
    "train_gp01.py"
    "make_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
