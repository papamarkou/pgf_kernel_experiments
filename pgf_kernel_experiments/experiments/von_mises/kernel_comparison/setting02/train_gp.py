# %% Import packages

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

# from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting02.set_paths import data_path, output_path
from pgf_kernel_experiments.runners import ExactSingleGPRunner

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)

grid = data[:, 1:3]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %% Get training data

train_pos = grid[train_ids, :]

train_output = z[train_ids]

# %% Get test data

test_pos = grid[test_ids, :]

test_output = z[test_ids]

# %% Plot training and test data

fontsize = 11

titles = ['von Mises density', 'Training data', 'Test data']

cols = ['green', 'orange', 'brown']

fig, ax = plt.subplots(1, 3, figsize=[12, 3], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.35,
    hspace=0.0
)

ax[0].plot(x, y, z, color=cols[0], lw=2)

ax[1].scatter(train_pos[:, 0], train_pos[:, 1], train_output, color=cols[1], s=2)

ax[2].scatter(test_pos[:, 0], test_pos[:, 1], test_output, color=cols[2], s=2)

for i in range(3):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color='black', lw=2, zorder=0)

    ax[i].grid(False)

    ax[i].tick_params(pad=-1.5)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 11))

    ax[i].set_title(titles[i], fontsize=fontsize, pad=-1.5)

    ax[i].set_xlabel('x', fontsize=fontsize, labelpad=-3)
    ax[i].set_ylabel('y', fontsize=fontsize, labelpad=-3)
    ax[i].set_zlabel('z', fontsize=fontsize, labelpad=-27)

    ax[i].set_xticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=fontsize)

    ax[i].zaxis.set_rotate_label(False)

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output.T, dtype=torch.float64)

# %% Set up ExactSingleGPRunner

# kernel = GFKernel(width=[20, 20, 20])
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
# kernel = gpytorch.kernels.PeriodicKernel()
# kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)

runner = ExactSingleGPRunner(train_x, train_y, kernel)

# %% Set the model in double mode

runner.model.double()
runner.model.likelihood.double()

# %% Configurate training setup for GP model

optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.1)

num_iters = 50

# %% Train GP model to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizer, num_iters)

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

# %% Plot data and predictions

fontsize = 11

titles = ['von Mises density', 'Training data', 'Test data', 'Predictions']

cols = ['green', 'orange', 'brown', 'blue']

fig, ax = plt.subplots(1, 4, figsize=[14, 3], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.25,
    hspace=0.0
)

ax[0].plot(x, y, z, color=cols[0], lw=2)

ax[1].scatter(train_pos[:, 0], train_pos[:, 1], train_output, color=cols[1], s=2)

ax[2].scatter(test_pos[:, 0], test_pos[:, 1], test_output, color=cols[2], s=2)

ax[3].scatter(test_pos[:, 0], test_pos[:, 1], predictions.mean, color=cols[3], s=2)

for i in range(4):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color='black', lw=2, zorder=0)

    ax[i].grid(False)

    ax[i].tick_params(pad=-1.5)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 11))

    ax[i].set_title(titles[i], fontsize=fontsize, pad=-1.5)

    ax[i].set_xlabel('x', fontsize=fontsize, labelpad=-3)
    ax[i].set_ylabel('y', fontsize=fontsize, labelpad=-3)
    ax[i].set_zlabel('z', fontsize=fontsize, labelpad=-27)

    ax[i].set_xticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=fontsize)

    ax[i].zaxis.set_rotate_label(False)
