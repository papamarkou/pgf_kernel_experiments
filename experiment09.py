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

dataset['lat'].shape, dataset['lon'].shape, dataset['tempanomaly'].shape
# ((90,), (180,), (1711, 90, 180))

# %%

# from pyproj import CRS

# %%

import numpy as np

lon, lat = np.meshgrid(dataset['lon'], dataset['lat'])

# %%

import matplotlib.pyplot as plt

plt.pcolormesh(lon, lat, dataset['tempanomaly'][1695, :, :], cmap="RdBu_r")
plt.colorbar()

# %%

tempanomaly_nans = np.isnan(dataset['tempanomaly'][1695, :, :])

# lon_nans, lat_nans = np.meshgrid(lon[tempanomaly_nans], lat[tempanomaly_nans])

tempanomaly_no_nans = ~np.isnan(dataset['tempanomaly'][1695, :, :])

# lon_no_nans, lat_no_nans = np.meshgrid(lon[tempanomaly_no_nans], lat[tempanomaly_no_nans])

# %%

from scipy.interpolate import griddata

gdata = griddata(
    np.column_stack((lon[tempanomaly_no_nans].flatten(), lat[tempanomaly_no_nans].flatten())),
    dataset['tempanomaly'][1695, :, :][tempanomaly_no_nans],
    np.column_stack((lon[tempanomaly_nans].flatten(), lat[tempanomaly_nans].flatten())),
    method='cubic'
)

dataset['tempanomaly'][1695, :, :][tempanomaly_nans] = gdata

# %%

from scipy.interpolate import CloughTocher2DInterpolator

interp = CloughTocher2DInterpolator(list(zip(lon, lat)), dataset['tempanomaly'][1695, :, :])
z = interp(lon, lat)

# %%

from cartopy import crs as ccrs

projMoll = ccrs.Mollweide(central_longitude=0)

fig = plt.figure(figsize=(11, 8.5))

ax = plt.subplot(1, 1, 1, projection=projMoll)
ax.coastlines()

dataplot = ax.pcolormesh(lon, lat, dataset['tempanomaly'][1695, :, :], cmap="RdBu_r", transform=ccrs.PlateCarree())

plt.title('Temperature anomaly')

plt.colorbar(dataplot, orientation='horizontal');

# %%

from datetime import date, timedelta

start_date = date.fromisoformat('1800-01-01')

start_date + timedelta(days=int(dataset['time'][1695]))

# %%

import numpy as np

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:,0], points[:,1])

from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()

# %%
