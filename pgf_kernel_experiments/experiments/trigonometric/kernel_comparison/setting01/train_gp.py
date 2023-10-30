# %% Import packages

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting01.set_env import data_path
from pgf_kernel_experiments.runners import ExactSingleGPRunner

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

title_fontsize = 15
axis_fontsize = 11

titles = ['von Mises density', 'Training data', 'Test data']

fig, ax = plt.subplots(1, 3, figsize=[12, 3], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.35,
    hspace=0.0
)

line_width = 2

# https://matplotlib.org/stable/tutorials/colors/colors.html

pdf_line_col = '#069AF3' # azure
circle_line_col = 'black'

train_point_col = '#F97306' # orange
test_point_col = '#C20078' # magenta

# https://matplotlib.org/stable/api/markers_api.html

point_marker = 'o'

point_size = 8

ax[0].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[1].scatter(
    train_pos[:, 0],
    train_pos[:, 1],
    train_output,
    color=train_point_col,
    marker=point_marker,
    s=point_size
)

ax[1].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[2].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    test_output,
    color=test_point_col,
    marker=point_marker,
    s=point_size
)

ax[2].plot(x, y, z, color=pdf_line_col, lw=line_width)

for i in range(3):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color=circle_line_col, lw=line_width, zorder=0)

    ax[i].grid(False)

    ax[i].tick_params(pad=-1.5)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 11))

    ax[i].set_title(titles[i], fontsize=title_fontsize, pad=-1.5)

    ax[i].set_xlabel('x', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_ylabel('y', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_zlabel('z', fontsize=axis_fontsize, labelpad=-27)

    ax[i].set_xticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=axis_fontsize)

    ax[i].zaxis.set_rotate_label(False)

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output.T, dtype=torch.float64)

# %% Set up ExactSingleGPRunner

kernel = GFKernel(width=[30, 30, 30])
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
# kernel = gpytorch.kernels.PeriodicKernel()
# kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)

runner = ExactSingleGPRunner(train_x, train_y, kernel)

# %% Set the model in double mode

runner.model.double()
runner.model.likelihood.double()

# %% Configure training setup for GP model

# list(runner.model.named_parameters())

# optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.7)

# optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.7, betas=(0.9, 0.99))

optimizer = torch.optim.Adam([
    {"params": runner.model.likelihood.noise_covar.raw_noise, "lr": 0.8},
    {"params": runner.model.mean_module.raw_constant, "lr": 0.5},
    {"params": runner.model.covar_module.pars0, "lr": 0.9},
    {"params": runner.model.covar_module.pars1, "lr": 0.9},
    {"params": runner.model.covar_module.pars2, "lr": 0.9}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.05)

# scheduler = None

num_iters = 100

# %% Train GP model to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizer, num_iters, scheduler=scheduler)

list(runner.model.named_parameters())

# %% Make predictions

predictions = runner.test(test_x)

# %% Compute error metrics

scores = runner.assess(
    predictions,
    test_y,
    metrics=[
        gpytorch.metrics.mean_absolute_error,
        gpytorch.metrics.mean_squared_error,
        lambda predictions, y : gpytorch.metrics.negative_log_predictive_density(predictions, y)
    ]
)

# %% Plot data and predictions

titles = ['von Mises density', 'Training data', 'Test data', 'Predictions']

fig, ax = plt.subplots(1, 4, figsize=[14, 3], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.3,
    hspace=0.0
)

pred_point_col = '#E50000' # red

ax[0].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[1].scatter(
    train_pos[:, 0],
    train_pos[:, 1],
    train_output,
    color=train_point_col,
    marker=point_marker,
    s=point_size
)

ax[1].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[2].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    test_output,
    color=test_point_col,
    marker=point_marker,
    s=point_size
)

ax[2].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[3].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions.mean,
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[3].plot(x, y, z, color=pdf_line_col, lw=line_width)

for i in range(4):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color=circle_line_col, lw=line_width, zorder=0)

    ax[i].grid(False)

    ax[i].tick_params(pad=-1.5)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 11))

    ax[i].set_title(titles[i], fontsize=title_fontsize, pad=-1.5)

    ax[i].set_xlabel('x', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_ylabel('y', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_zlabel('z', fontsize=axis_fontsize, labelpad=-27)

    ax[i].set_xticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=axis_fontsize)

    ax[i].zaxis.set_rotate_label(False)
