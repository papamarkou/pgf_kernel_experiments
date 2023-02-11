```
# conda remove --name dedalus --all

conda create -n dedalus python=3.10

conda activate dedalus

conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1

conda install -c conda-forge dedalus c-compiler "h5py=*=mpi*"

conda uninstall --force dedalus

CC=mpicc pip3 install --no-cache http://github.com/dedalusproject/dedalus/zipball/master/

# conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn ipykernel

conda install -c conda-forge ipykernel
```
