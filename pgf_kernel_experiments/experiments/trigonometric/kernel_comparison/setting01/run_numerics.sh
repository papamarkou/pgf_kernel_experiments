#!/bin/bash

declare -a files=(
    "simulate_data.py"
    "train_pgf_gp.py"
    "train_rbf_gp.py"
    "make_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
