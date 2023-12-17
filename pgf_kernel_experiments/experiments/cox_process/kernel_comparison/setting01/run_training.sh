#!/bin/bash

declare -a files=(
    "train_pgf_dkl.py"
    "train_rbf_dkl.py"
    "train_matern_dkl.py"
    "train_periodic_dkl.py"
)

for file in "${files[@]}"
do
   echo "Executing $file..."
   python $file
done
