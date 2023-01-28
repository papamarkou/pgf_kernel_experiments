# %% Download data

# 1. Go to https://www.ncei.noaa.gov/maps/grid-extract/
# 2. Pick ETOPO_2022 Hillshade (Ice Surface; 15 arcseconds)
# 3. Click on button 'Select area of interest by rectangle'
# 4. Click on button 'Download Data'

# %% Import packages

import numpy as np

from geotiff import GeoTiff

# %% Load data from TIFF file

geo_tiff = GeoTiff("exportImage3.tiff")

np.array(geo_tiff.read())

# %% Standardize data or convert from (371, 519, 3) to (371, 519)

# For now greyscale
# https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale

# https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/index.html

# %% Plot data

# %% Pick a stripe as test set, keeping rest as training

# %%
