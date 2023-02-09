# %% Import packages

import gpytorch
import torch
import zarr

import matplotlib.pyplot as plt
import numpy as np

from cartopy import crs
from datetime import date, timedelta
from scipy.interpolate import griddata

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

# %% Load data

dataset = zarr.load('gistemp1200_GHCNv4_ERSSTv5.zarr')

n_times, n_lat, n_lon = dataset['tempanomaly'].shape

n_locs = n_lat * n_lon

# %% Select a time point

t = 1695

start_date = date.fromisoformat('1800-01-01')

current_date = start_date + timedelta(days=int(dataset['time'][t]))

# %% Compute Cartesian coordinates from polar coordinates

lon, lat = np.meshgrid(dataset['lon'], dataset['lat'])

x, y, z = cartesian_from_latlon(lon, lat)

x = x / np.linalg.norm(x, ord=2)

y = y / np.linalg.norm(y, ord=2)

z = z / np.linalg.norm(z, ord=2)

pos = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

# %% Plot original data at chosen time point without projection (before interpolation)

fontsize = 11

plt.figure(figsize=(10, 5))

plt.pcolormesh(lon, lat, dataset['tempanomaly'][t, :, :], cmap='RdBu_r')

plt.title('Temperature anomaly, {} (before interpolation)'.format(current_date), fontsize=fontsize)

plt.xticks(np.linspace(-150, 150, num=7), fontsize=fontsize)

plt.yticks(np.linspace(-80, 80, num=5), fontsize=fontsize)

cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=fontsize)

# %% Plot original data at chosen time point in Mollweide projection (before interpolation)

fontsize = 11

plt.figure(figsize=(11, 9))

ax = plt.subplot(1, 1, 1, projection=crs.Mollweide(central_longitude=0))

ax.coastlines()

dataplot = ax.pcolormesh(lon, lat, dataset['tempanomaly'][t, :, :], cmap='RdBu_r', transform=crs.PlateCarree())

plt.title('Temperature anomaly, {} (before interpolation)'.format(current_date), fontsize=fontsize)

cbar = plt.colorbar(dataplot, orientation='horizontal')

cbar.ax.tick_params(labelsize=fontsize)

# %% Interpolate missing values

tempanomaly_nans = np.isnan(dataset['tempanomaly'][t, :, :])

# lon_nans, lat_nans = np.meshgrid(lon[tempanomaly_nans], lat[tempanomaly_nans])

tempanomaly_no_nans = ~np.isnan(dataset['tempanomaly'][t, :, :])

# lon_no_nans, lat_no_nans = np.meshgrid(lon[tempanomaly_no_nans], lat[tempanomaly_no_nans])

gdata = griddata(
    np.column_stack((lon[tempanomaly_no_nans].flatten(), lat[tempanomaly_no_nans].flatten())),
    dataset['tempanomaly'][1695, :, :][tempanomaly_no_nans],
    np.column_stack((lon[tempanomaly_nans].flatten(), lat[tempanomaly_nans].flatten())),
    method='cubic'
)

dataset['tempanomaly'][t, :, :][tempanomaly_nans] = gdata

# %% Plot original data at chosen time point without projection (after interpolation)

fontsize = 11

plt.figure(figsize=(10, 5))

plt.pcolormesh(lon, lat, dataset['tempanomaly'][t, :, :], cmap='RdBu_r')

plt.title('Temperature anomaly, {} (after interpolation)'.format(current_date), fontsize=fontsize)

plt.xticks(np.linspace(-150, 150, num=7), fontsize=fontsize)

plt.yticks(np.linspace(-80, 80, num=5), fontsize=fontsize)

cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=fontsize)

# %% Plot original data at chosen time point in Mollweide projection (after interpolation)

fontsize = 11

plt.figure(figsize=(11, 9))

ax = plt.subplot(1, 1, 1, projection=crs.Mollweide(central_longitude=0))

ax.coastlines()

dataplot = ax.pcolormesh(lon, lat, dataset['tempanomaly'][t, :, :], cmap='RdBu_r', transform=crs.PlateCarree())

plt.title('Temperature anomaly, {} (after interpolation)'.format(current_date), fontsize=fontsize)

cbar = plt.colorbar(dataplot, orientation='horizontal')

cbar.ax.tick_params(labelsize=fontsize)

# %% Generate training data

ids = np.arange(n_locs)

n_train = int(0.7 * n_locs)

train_ids = np.random.RandomState(4).choice(ids, size=n_train, replace=False)

train_ids.sort()

train_pos = pos[train_ids, :]

train_output = dataset['tempanomaly'][t, :, :].flatten()[[train_ids]].squeeze()

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

n_test = n_locs - n_train

test_pos = pos[test_ids, :]

test_output = dataset['tempanomaly'][t, :, :].flatten()[[test_ids]].squeeze()

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output, dtype=torch.float64)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20, 20, 20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    # gpytorch.kernels.PeriodicKernel(),
    # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
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
