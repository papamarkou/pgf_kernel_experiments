```
# conda remove --name uai2023 --all

conda create -n uai2023 python=3.10

conda activate uai2023

# Related to shell convection and shallow water examples

conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1

conda install -c conda-forge numpy pytorch-cpu torchvision gpytorch

conda install -c conda-forge scipy pandas scikit-learn matplotlib seaborn ipykernel

# Related to Bloch density example

conda install -c conda-forge qutip

# Related to shell convection and shallow water examples

conda install -c conda-forge dedalus c-compiler "h5py=*=mpi*"

conda uninstall --force dedalus

CC=mpicc pip3 install --no-cache http://github.com/dedalusproject/dedalus/zipball/master/

cd /home/theodore/opt/python/packages

git clone git@github.com:papamarkou/pgfml.git

pip install -e pgfml -r pgfml/requirements.txt

git clone git@github.com:papamarkou/pgf_kernel_experiments.git

pip install -e pgf_kernel_experiments -r pgf_kernel_experiments/requirements.txt
```
