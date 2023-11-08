#!/bin/bash

declare -a files=(
    "run_data_simulation.sh"
    "run_training.sh"
    "run_predictions.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   ./$file
done
