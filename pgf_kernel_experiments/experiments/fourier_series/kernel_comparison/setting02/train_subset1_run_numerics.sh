#!/bin/bash

declare -a files=(
    "train_subset1_train_gps.py"
    "train_subset1_make_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
