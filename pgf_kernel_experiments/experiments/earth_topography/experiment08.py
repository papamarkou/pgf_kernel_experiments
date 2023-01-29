# %% Download data

# 1. Go to https://www.ncei.noaa.gov/maps/grid-extract/
# 2. Pick ETOPO_2022 Hillshade (Ice Surface; 15 arcseconds)
# 3. Click on button 'Select area of interest by rectangle'
# 4. Click on button 'Download Data'

# Longitude: west: 18.5, east: 21.5
# Latitude: south: 34.0, north: 37.0

# %% Import packages

import numpy as np

from geotiff import GeoTiff

# %% Load data from TIFF file

geo_tiff = GeoTiff("sea_bed.tiff")

dataset = np.array(geo_tiff.read())

# %% Plot original data

import matplotlib.pyplot as plt

plt.imshow(dataset)

# %% Convert from (371, 519, 3) to (371, 519)

# For now grayscale
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
# https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale

# https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/index.html

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

gdataset = rgb2gray(dataset)

# %% Plot grayscale data

import matplotlib.pyplot as plt

plt.imshow(gdataset, cmap=plt.get_cmap('gray'))

# %% Pick a stripe as test set, keeping rest as training

# %%
