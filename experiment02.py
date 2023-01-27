# %% Import packages

import gpytorch
import torch

import gstools as gs
import numpy as np

from matplotlib import pyplot as plt

from pgfml.kernels import GFKernel

from exact_single_gp_runner import ExactSingleGPRunner

# %% Simulate data

# See
# https://buildmedia.readthedocs.org/media/pdf/gstools/latest/gstools.pdf
# Section 2.10 (normalizing data)

x = y = np.linspace(-30, 30, num=60)

grid = gs.generate_grid([x, y])

grid_normed = grid / np.linalg.norm(grid, axis=0)

model = gs.Gaussian(dim=2, var=1, len_scale=10)

srf = gs.SRF(model, seed=200)

srf_normed = gs.SRF(model, seed=200)

# Generate the original field
srf(grid)

# Generate the normed field
srf_normed(grid_normed)

# %% Generate training data

ids = np.arange(srf_normed.field.size)

n_train = int(0.75 * srf_normed.field.size)

train_samples = np.random.RandomState(201).choice(ids, size=n_train, replace=False)

train_pos = grid_normed[:, train_samples]
train_labels = srf_normed.field[train_samples]

# %% Generate test data

test_samples = np.array(list(set(ids).difference(set(train_samples))))

n_test = srf_normed.field.size - n_train

test_pos = grid_normed[:, test_samples]
test_labels = srf_normed.field[test_samples]

# %% Plot training and test data

fig, ax = plt.subplots(1, 4, figsize=[12, 3])

ax[0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[2].scatter(*train_pos, c=train_labels)
ax[3].scatter(*test_pos, c=test_labels)

ax[0].set_title("Original field")
ax[1].set_title("Normed field")
ax[2].set_title("Training data")
ax[3].set_title("Test data")

[ax[i].set_aspect("equal") for i in range(4)]

# %% Set training and test data in GPyTorch-friendly format

train_x = torch.as_tensor(train_pos.T, dtype=torch.float32)
train_y = torch.as_tensor(train_labels.T, dtype=torch.float32)

test_x = torch.as_tensor(test_pos.T, dtype=torch.float32)
test_y = torch.as_tensor(test_labels.T, dtype=torch.float32)

# %% Set up single GP runner

runner = ExactSingleGPRunner(train_x, train_y, gpytorch.kernels.RBFKernel())
# runner = ExactSingleGPRunner(train_x, train_y, gpytorch.kernels.ScaleKernel(GFKernel(width=[20])))

# %% Print parameter names and values

for param_name, param_value in runner.model.named_parameters():
    print('Parameter name: {}. Parameter value: {}'.format(param_name, param_value))

# %% Configurate training setup for GP model

# Set the optimizer
optimizer = torch.optim.SGD(runner.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# Set number of training itereations
num_iters = 20

# %% Train GP model to find optimal hyperparameters

losses = runner.train(optimizer, train_x, train_y, num_iters)

# %% Make predictions

predictions = runner.test(test_x)

# %% Compute error metrics

print('Test MAE: {}'.format(gpytorch.metrics.mean_absolute_error(predictions, test_y)))

# %% Plot predictions

fig, ax = plt.subplots(1, 5, figsize=[16, 3])

ax[0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[2].scatter(*train_pos, c=train_labels)
ax[3].scatter(*test_pos, c=test_labels)
ax[4].scatter(*test_pos, c=predictions.mean.numpy())

ax[0].set_title("Original field")
ax[1].set_title("Normed field")
ax[2].set_title("Training data")
ax[3].set_title("Test data")
ax[4].set_title("Predictions")

[ax[i].set_aspect("equal") for i in range(5)]
