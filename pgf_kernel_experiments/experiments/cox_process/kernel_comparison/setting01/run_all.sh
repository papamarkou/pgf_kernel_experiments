#!/bin/bash

declare -a files=(
    "run_numerics.sh"
    "run_plots.sh"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   ./$file
done
