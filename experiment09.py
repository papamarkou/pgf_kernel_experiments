# %%

# Data downloaded from this page:
# https://data.giss.nasa.gov/gistemp/

# Direct link to download the Zarr file:
# https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.zarr.tar.gz

# %%

import zarr

# %%

dataset = zarr.load('gistemp1200_GHCNv4_ERSSTv5.zarr')

# %%

dataset['lat']

dataset['lon']

dataset['tempanomaly']

dataset['time']

# %%

# from pyproj import CRS

# %%

import numpy as np

lon, lat = np.meshgrid(dataset['lon'], dataset['lat'])

# %%

import matplotlib.pyplot as plt

plt.contourf(lon, lat, dataset['tempanomaly'][1693, :, :], cmap="RdBu_r")
# plt.contourf(lon.transpose(), lat.transpose(), dataset['tempanomaly'][1693, :, :].transpose(), cmap="RdBu_r")
plt.colorbar()

# %%

from cartopy import crs as ccrs

projMoll = ccrs.Mollweide(central_longitude=0)

fig = plt.figure(figsize=(11, 8.5))
ax = plt.subplot(1, 1, 1, projection=projMoll)
ax.coastlines()
dataplot = ax.contourf(lon, lat, dataset['tempanomaly'][1695, :, :], cmap="RdBu_r", transform=ccrs.PlateCarree())
plt.colorbar(dataplot, orientation='horizontal');

# %%

dataset['lat'].shape, dataset['lon'].shape, dataset['tempanomaly'].shape
# ((90,), (180,), (1711, 90, 180))
