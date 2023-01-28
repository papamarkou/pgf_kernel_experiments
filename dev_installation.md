```
conda create -n pgfml python=3.10

conda activate pgfml

conda install -c conda-forge numpy pytorch-cpu torchvision gpytorch

conda install -c conda-forge pandas scikit-learn matplotlib seaborn ipykernel

conda install -c conda-forge zarr cartopy gstools

# conda install -c conda-forge geotiff python-geotiff iris iris-sample-data

cd /home/theodore/opt/python/packages

git clone git@github.com:papamarkou/pgfml.git

pip install -e pgfml -r pgfml/requirements.txt
```
