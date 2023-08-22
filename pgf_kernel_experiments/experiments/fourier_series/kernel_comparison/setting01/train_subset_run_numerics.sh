#!/bin/bash

declare -a files=(
    "train_subset_train_gps.py"
    "train_subset_make_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
