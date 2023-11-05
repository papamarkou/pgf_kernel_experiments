#!/bin/bash

declare -a files=(
    "train_pgf_gp.py"
    "train_rbf_gp.py"
    "train_matern_gp.py"
    "train_periodic_gp.py"
    "train_spectral_gp.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
