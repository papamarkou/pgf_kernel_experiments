#!/bin/bash

declare -a files=(
    "train_set_train_gps.py"
    "train_set_make_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
