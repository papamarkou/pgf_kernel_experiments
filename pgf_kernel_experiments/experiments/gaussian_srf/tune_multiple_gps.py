# %% Import packages

import gpytorch
import torch

import gstools as gs
import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.runners.exact_multi_gp_runner import ExactMultiGPRunner
from pgfml.kernels import GFKernel

# %% Simulate data

# See
# https://buildmedia.readthedocs.org/media/pdf/gstools/latest/gstools.pdf
# Section 2.10 (normalizing data)

x = y = np.linspace(-30, 30, num=60)

grid = gs.generate_grid([x, y])

grid_normed = grid / np.linalg.norm(grid, axis=0)

model = gs.Gaussian(dim=2, var=1, len_scale=10)

srf = gs.SRF(model, seed=4)

srf_normed = gs.SRF(model, seed=4)

# Generate the original field
srf(grid)

# Generate the normed field
srf_normed(grid_normed)

# %% Generate training data

ids = np.arange(srf_normed.field.size)

n_train = int(0.75 * srf_normed.field.size)

train_ids = np.random.RandomState(3).choice(ids, size=n_train, replace=False)

train_pos = grid_normed[:, train_ids]

train_output = srf_normed.field[train_ids]

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

n_test = srf_normed.field.size - n_train

test_pos = grid_normed[:, test_ids]

test_output = srf_normed.field[test_ids]

# %% Plot training and test data

fontsize = 11

fig, ax = plt.subplots(1, 4, figsize=[14, 3])

ax[0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[2].scatter(*train_pos, c=train_output)
ax[3].scatter(*test_pos, c=test_output)

ax[0].set_title(r'$Original~field$', fontsize=fontsize)
ax[1].set_title(r'$Normed~field$', fontsize=fontsize)
ax[2].set_title(r'$Training~data$', fontsize=fontsize)
ax[3].set_title(r'$Test~data$', fontsize=fontsize)

ax[0].set_xticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[1].set_xticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[2].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[3].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)

ax[0].set_yticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[1].set_yticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[2].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[3].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)

[ax[i].set_aspect("equal") for i in range(4)]

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos.T, dtype=torch.float32)
train_y = torch.as_tensor(train_output.T, dtype=torch.float32)

test_x = torch.as_tensor(test_pos.T, dtype=torch.float32)
test_y = torch.as_tensor(test_output.T, dtype=torch.float32)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
]

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

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

# %% Plot predictions

fig, ax = plt.subplots(1, 5, figsize=[16, 3])

ax[0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[2].scatter(*train_pos, c=train_output)
ax[3].scatter(*test_pos, c=test_output)
ax[4].scatter(*test_pos, c=predictions[1].mean.numpy())

ax[0].set_title("Original field")
ax[1].set_title("Normed field")
ax[2].set_title("Training data")
ax[3].set_title("Test data")
ax[4].set_title("Predictions")

[ax[i].set_aspect("equal") for i in range(5)]

# %% PLot predictions

fontsize = 11

fig, ax = plt.subplots(2, 4, figsize=[16, 8])

ax[0, 0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[0, 1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[0, 2].scatter(*train_pos, c=train_output)
ax[0, 3].scatter(*test_pos, c=test_output)
ax[1, 0].scatter(*test_pos, c=predictions[0].mean)
ax[1, 1].scatter(*test_pos, c=predictions[0].mean)
ax[1, 2].scatter(*test_pos, c=predictions[0].mean)
ax[1, 3].scatter(*test_pos, c=predictions[0].mean)

ax[0, 0].set_title(r'$Original~field$', fontsize=fontsize)
ax[0, 1].set_title(r'$Normed~field$', fontsize=fontsize)
ax[0, 2].set_title(r'$Training~data$', fontsize=fontsize)
ax[0, 3].set_title(r'$Test~data$', fontsize=fontsize)

ax[1, 0].set_title(r'$Test~data$', fontsize=fontsize)
ax[1, 1].set_title(r'$Test~data$', fontsize=fontsize)
ax[1, 2].set_title(r'$Test~data$', fontsize=fontsize)
ax[1, 3].set_title(r'$Test~data$', fontsize=fontsize)

ax[1, 0].set_title(r'$PGF~kernel$')
ax[1, 1].set_title(r'$RBF~kernel$')
ax[1, 2].set_title(r'$Mat\acute{e}rn~kernel$')
ax[1, 3].set_title(r'$Spectral~kernel$')

ax[0, 0].set_xticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[0, 1].set_xticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[0, 2].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[0, 3].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 0].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 1].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 2].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 3].set_xticks(np.linspace(-1, 1, num=5), fontsize=fontsize)

ax[0, 0].set_yticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[0, 1].set_yticks(np.linspace(0, 60, num=7), fontsize=fontsize)
ax[0, 2].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[0, 3].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 0].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 1].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 2].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)
ax[1, 3].set_yticks(np.linspace(-1, 1, num=5), fontsize=fontsize)

[ax[i, j].set_aspect("equal") for i in range(2) for j in range(4)]
