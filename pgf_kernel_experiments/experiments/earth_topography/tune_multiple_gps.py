# %% Import packages

import gpytorch
import torch

import matplotlib.pyplot as plt
import numpy as np

from geotiff import GeoTiff

from pgf_kernel_experiments.runners import ExactMultiGPRunner
from pgfml.kernels import GFKernel

# %% Function from converving from latitude-longtitude to Cartesian coordinates

# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
# https://en.wikipedia.org/wiki/Spherical_coordinate_system

def cartesian_from_latlon(lon, lat, radius=6371):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)

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

# %% Convert image from RGB to grayscale

gray_dataset = rgb2gray(rgb_dataset)

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

# theta is lat, phi is lon

lon_grid_flat = lon_grid.flatten().squeeze()

lat_grid_flat = lat_grid.flatten().squeeze()

x, y, z = cartesian_from_latlon(lon_grid_flat, lat_grid_flat, radius=1)

pos = np.column_stack((x, y, z))

# %% Generate training data

ids = np.arange(n_samples)

n_train = int(0.7 * n_samples)

train_ids = np.random.choice(ids, size=n_train, replace=False)

train_ids.sort()

train_pos = pos[train_ids, :]

train_output = gray_dataset.flatten()[[train_ids]].squeeze()

# %% Generate test datat

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

n_test = n_samples - n_train

test_pos = pos[test_ids, :]

test_output = gray_dataset.flatten()[[test_ids]].squeeze()

# %% Plot training and test data

fontsize = 11

rotation = 60

fig, ax = plt.subplots(1, 4, figsize=[14, 3], sharey=True)

#  https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/

plt.subplots_adjust(wspace=0.2)

ax[0].imshow(rgb_dataset)

ax[1].imshow(gray_dataset, cmap=plt.get_cmap('gray'))

gray_train_dataset_vis = gray_dataset.copy()
gray_train_dataset_vis = gray_train_dataset_vis.flatten()
gray_train_dataset_vis[[test_ids]] = gray_dataset.max()
gray_train_dataset_vis = gray_train_dataset_vis.reshape(gray_dataset.shape)

ax[2].imshow(gray_train_dataset_vis, cmap=plt.get_cmap('gray'))

gray_test_dataset_vis = gray_dataset.copy()
gray_test_dataset_vis = gray_test_dataset_vis.flatten()
gray_test_dataset_vis[[train_ids]] = gray_dataset.max()
gray_test_dataset_vis = gray_test_dataset_vis.reshape(gray_dataset.shape)

ax[3].imshow(gray_test_dataset_vis, cmap=plt.get_cmap('gray'))

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

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output, dtype=torch.float64)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20, 20, 20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    gpytorch.kernels.PeriodicKernel(),
    gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=3)
]

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Set the models in double mode

for i in range(len(kernels)):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

# %% Configurate training setup for GP models

optimizers = []

for i in range(runner.num_gps()):
    optimizers.append(torch.optim.Adam(runner.single_runners[i].model.parameters(), lr=0.1))

n_iters = 10

# %% Train GP models to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizers, n_iters)

# %% Make predictions

predictions = runner.test(test_x)

# %% Compute error metrics

scores = runner.assess(
    predictions,
    test_y,
    metrics=[
        gpytorch.metrics.mean_absolute_error,
        gpytorch.metrics.mean_squared_error,
        lambda predictions, y : -gpytorch.metrics.negative_log_predictive_density(predictions, y)
    ]
)
