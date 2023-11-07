#!/bin/bash

declare -a files=(
    "make_pgf_gp_predictions.py"
    "make_rbf_gp_predictions.py"
    "make_matern_gp_predictions.py"
    "make_periodic_gp_predictions.py"
    "make_spectral_gp_predictions.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
