```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

sudo apt-get --purge remove
sudo apt-get --purge autoremove

# https://gist.github.com/verazuo/19f381e4e2e546a9edcf66fc103d24a4
# https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5

sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get --purge autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

sudo apt-get update
sudo apt-get upgrade

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

sudo apt-get install libnvidia-common-525
sudo apt-get install libnvidia-gl-525
sudo apt-get install nvidia-driver-525

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

sudo apt-get install cuda-toolkit-11-7

echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar -xvf ${CUDNN_TAR_FILE}

sudo cp -P cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/cudnn.h /usr/local/cuda-11.7/include
sudo cp -P cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/lib64/libcudnn*

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
mkdir -p /home/theodore/opt/continuum/miniconda
chmod u+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# /home/theodore/opt/continuum/miniconda/miniconda3

conda config --set auto_activate_base false

conda update conda
conda update --all

conda create -n aistats python=3.10

conda activate aistats

# Related to shell convection and shallow water examples

conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# conda install -c conda-forge numpy pytorch-cpu torchvision gpytorch

conda install -c conda-forge scipy pandas scikit-learn matplotlib seaborn ipykernel

# python
# import torch
# torch.cuda.is_available()
# torch.cuda.device_count()
# torch.cuda.get_device_name(0)

# Related to Bloch density example

conda install -c conda-forge qutip

# Related to earth topography example

conda install -c conda-forge geotiff

# Related to shell convection and shallow water examples

conda install -c conda-forge dedalus c-compiler "h5py=*=mpi*"

conda uninstall --force dedalus

CC=mpicc pip3 install --no-cache http://github.com/dedalusproject/dedalus/zipball/master/

mkdir -p /home/theodore/opt/python/packages
cd /home/theodore/opt/python/packages

git clone git@github.com:papamarkou/pgfml.git

pip install -e pgfml -r pgfml/requirements.txt

git clone git@github.com:papamarkou/pgf_kernel_experiments.git

pip install -e pgf_kernel_experiments -r pgf_kernel_experiments/requirements.txt
```
