# %% Import packages

import gpytorch
import torch

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import vonmises

from pgf_kernel_experiments.runners import ExactMultiGPRunner
from pgfml.kernels import GFKernel

# %% Generate data

n_samples = 3000

theta = np.linspace(-np.pi, np.pi, num=n_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

grid = np.stack((x, y))

z = vonmises.pdf(theta, kappa=2., loc=0., scale=0.05)

# %% Generate training data

ids = np.arange(n_samples)

n_train = int(0.7 * n_samples)

train_ids = np.random.RandomState(300).choice(ids, size=n_train, replace=False)

train_pos = grid[:, train_ids]

train_output = z[train_ids]

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

n_test = n_samples - n_train

test_pos = grid[:, test_ids]

test_output = z[test_ids]

# %% Plot training and test data

fontsize = 11

titles = ['von Mises', 'Training data', 'Test data']

titles = [r'$von~Mises~density$', r'$Training~data$', r'$Test~data$']

cols = ['green', 'orange', 'brown']

fig, ax = plt.subplots(1, 3, figsize=[12, 3], subplot_kw={'projection': '3d'})

ax[0].plot(x, y, z, color=cols[0], lw=2)

ax[1].scatter(train_pos[0, :], train_pos[1, :], train_output, color=cols[1], s=2)

ax[2].scatter(test_pos[0, :], test_pos[1, :], test_output, color=cols[2], s=2)

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
    ax[i].set_zlabel('z', fontsize=fontsize, labelpad=-3)

    ax[i].set_xticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=fontsize)

    ax[i].zaxis.set_rotate_label(False)

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos.T, dtype=torch.float32)
train_y = torch.as_tensor(train_output.T, dtype=torch.float32)

test_x = torch.as_tensor(test_pos.T, dtype=torch.float32)
test_y = torch.as_tensor(test_output.T, dtype=torch.float32)

# %% Set up ExactMultiGPRunner

# kernels = [
#     # gpytorch.kernels.ScaleKernel(GFKernel(width=[20, 20, 20])),
#     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
#     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
#     gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
#     gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
#     gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
# ]

kernels = [
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
]

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Configurate training setup for GP models

optimizers = []

for i in range(runner.num_gps()):
    optimizers.append(torch.optim.SGD(runner.single_runners[i].model.parameters(), lr=0.01))

n_iters = 20

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
        lambda predictions, test_y : -gpytorch.metrics.negative_log_predictive_density(predictions, test_y)
    ]
)

# %% Plot predictions

fontsize = 11

titles = [
    [r'$von~Mises~density$', r'$Training~data$', r'$Test~data$', r'$PGF~kernel$'],
    [r'$RBF~kernel$', r'$Mat\acute{e}rn~kernel$', r'$Periodic~kernel$', r'$Spectral~kernel$']
]

cols = ['green', 'orange', 'brown', 'red', 'blue']

fig, ax = plt.subplots(2, 4, figsize=[16, 6], subplot_kw={'projection': '3d'})

ax[0, 0].plot(x, y, z, color=cols[0], lw=2)

ax[0, 1].scatter(train_pos[0, :], train_pos[1, :], train_output, color=cols[1], s=2)

ax[0, 2].scatter(test_pos[0, :], test_pos[1, :], test_output, color=cols[2], s=2)

ax[0, 3].scatter(test_pos[0, :], test_pos[1, :], predictions[0].mean, color=cols[3], s=2)

ax[1, 0].scatter(test_pos[0, :], test_pos[1, :], predictions[1].mean, color=cols[4], s=2)

ax[1, 1].scatter(test_pos[0, :], test_pos[1, :], predictions[2].mean, color=cols[4], s=2)

ax[1, 2].scatter(test_pos[0, :], test_pos[1, :], predictions[3].mean, color=cols[4], s=2)

ax[1, 3].scatter(test_pos[0, :], test_pos[1, :], predictions[4].mean, color=cols[4], s=2)

for i in range(2):
    for j in range(4):
        ax[i, j].set_proj_type('ortho')

        ax[i, j].plot(x, y, 0, color='black', lw=2, zorder=0)

        ax[i, j].grid(False)

        ax[i, j].tick_params(pad=-1.5)
        
        ax[i, j].set_xlim((-1, 1))
        ax[i, j].set_ylim((-1, 1))
        ax[i, j].set_zlim((0, 11))

        ax[i, j].set_title(titles[i][j], fontsize=fontsize, pad=-1.5)

        ax[i, j].set_xlabel('x', fontsize=fontsize, labelpad=-3)
        ax[i, j].set_ylabel('y', fontsize=fontsize, labelpad=-3)
        ax[i, j].set_zlabel('z', fontsize=fontsize, labelpad=-3)

        ax[i, j].set_xticks([-1, 0, 1], fontsize=fontsize)
        ax[i, j].set_yticks([-1, 0, 1], fontsize=fontsize)
        ax[i, j].set_zticks([0, 5., 10.], fontsize=fontsize)

        ax[i, j].zaxis.set_rotate_label(False)

# %%
