# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from geotiff import GeoTiff

# %% Function from converving from latitude-longtitude to Cartesian coordinates

# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
# https://en.wikipedia.org/wiki/Spherical_coordinate_system

def cartesian_from_latlon(lat, lon, radius=6371):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    return x, y, z

# %% Function for converting image from RGB to grayscale 

# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
# https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# %% Load data

geo_tiff = GeoTiff("sea_bed.tiff")

rgb_dataset = np.array(geo_tiff.read())

n_samples = rgb_dataset[:, :, 0].size

# %% Set up latitude-longitude information

n_lat, n_lon, _ = rgb_dataset.shape

lon_left = 18.5
lon_right = 21.5

lat_left = 34.0
lat_right = 37.0

lon = np.linspace(lon_left, lon_right, num=n_lat)
lat = np.linspace(lat_left, lat_right, num=n_lon)

lon_grid, lat_grid = np.meshgrid(lon, np.flip(lat))

# %% Compute Cartesian coordinates from polar coordinates

# theta: lat
# phi: lon

# %% Generate training data

ids = np.arange(n_samples)

n_train = int(0.7 * n_samples)

train_ids = np.random.RandomState(4).choice(ids, size=n_train, replace=False)

train_ids.sort()

# train_pos = grid_normed[:, train_ids]

# train_output = srf_normed.field[train_ids]

# %% Genearate test datat

test_ids = np.array(list(set(ids).difference(set(train_ids))))

n_test = n_samples - n_train

# test_pos = grid_normed[:, test_ids]

# test_output = srf_normed.field[test_ids]

# %% Convert image from RGB to grayscale

gray_dataset = rgb2gray(rgb_dataset)

# %% Plot training and test data

fontsize = 11

rotation = 60

fig, ax = plt.subplots(1, 4, figsize=[14, 3], sharey=True)

#  https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/

plt.subplots_adjust(wspace=0.2)

ax[0].imshow(rgb_dataset)
ax[1].imshow(gray_dataset, cmap=plt.get_cmap('gray'))
ax[2].imshow(gray_dataset, cmap=plt.get_cmap('gray'))
ax[3].imshow(gray_dataset, cmap=plt.get_cmap('gray'))

ax[0].set_title(r'$Original~image$', fontsize=fontsize)
ax[1].set_title(r'$Grayscale~image$', fontsize=fontsize)
ax[2].set_title(r'$Training~data$', fontsize=fontsize)
ax[3].set_title(r'$Test~data$', fontsize=fontsize)

xtick_ticks = np.linspace(0, 720, num=3)
xtick_labels = np.linspace(lon_left, lon_right, num=3)

ytick_ticks = np.linspace(0, 720, num=3)
ytick_labels = np.flip(np.linspace(lat_left, lat_right, num=3))

ax[0].set_xticks(ticks=xtick_ticks, labels=xtick_labels, fontsize=fontsize, rotation=rotation)
ax[1].set_xticks(ticks=xtick_ticks, labels=xtick_labels, fontsize=fontsize, rotation=rotation)
ax[2].set_xticks(ticks=xtick_ticks, labels=xtick_labels, fontsize=fontsize, rotation=rotation)
ax[3].set_xticks(ticks=xtick_ticks, labels=xtick_labels, fontsize=fontsize, rotation=rotation)

ax[0].set_yticks(ticks=ytick_ticks, labels=ytick_labels, fontsize=fontsize)
ax[1].set_yticks(ticks=ytick_ticks, labels=ytick_labels, fontsize=fontsize)
ax[2].set_yticks(ticks=ytick_ticks, labels=ytick_labels, fontsize=fontsize)
ax[3].set_yticks(ticks=ytick_ticks, labels=ytick_labels, fontsize=fontsize)

# %%

lon_grid, lat_grid = np.meshgrid(lon, np.flip(lat))

# %%
