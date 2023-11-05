#!/bin/bash

declare -a files=(
    "run_data_simulation.sh"
    "run_training.sh"
    "run_predictions.sh"
)

echo "Executing $file..."
./$file
