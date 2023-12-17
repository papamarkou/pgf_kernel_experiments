#!/bin/bash

declare -a files=(
    "make_pgf_dkl_predictions.py"
    "make_rbf_dkl_predictions.py"
    "make_matern_dkl_predictions.py"
    "make_periodic_dkl_predictions.py"
    "merge_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
