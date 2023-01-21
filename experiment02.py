# %% Import packages

import gpytorch
import torch

import gstools as gs
import numpy as np

from matplotlib import pyplot as plt

from pgfml.kernels import GFKernel

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

# %% Define GP model for exact inference

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = GFKernel(num_p=[20])
        self.covar_module = gpytorch.kernels.ScaleKernel(GFKernel(num_p=[20]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# %% Initialize likelihood and model

likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = ExactGPModel(train_x, train_y, likelihood)

# %% Print parameter names and values

for param_name, param_value in model.named_parameters():
    print('Parameter name: {}. Parameter value: {}'.format(param_name, param_value))

# %% Configurate training setup for GP model

num_training_iters = 20

# Get to training mode
model.train()
likelihood.train()

# model.covar_module.base_kernel.lengthscale

# Set the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# Set the loss for GPs to be the marginal log-likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# %% Train GP model to find optimal hyperparameters

for i in range(num_training_iters):
    # Zero gradients from previous iteration
    optimizer.zero_grad()

    # Output from model
    output = model(train_x)

    # Calculate loss and gradients
    loss = -mll(output, train_y)
    loss.backward()

    print('Iteration {}/{}, loss: {:.4f}'.format(i + 1, num_training_iters, loss.item()))

    optimizer.step()

# %% Make predictions

# Get to evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions by feeding model through likelihood
    preds = likelihood(model(test_x))

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))

# %% Plot predictions

fig, ax = plt.subplots(1, 5, figsize=[16, 3])

ax[0].imshow(srf.field.reshape(len(x), len(y)).T, origin="lower")
ax[1].imshow(srf_normed.field.reshape(len(x), len(y)).T, origin="lower")
ax[2].scatter(*train_pos, c=train_labels)
ax[3].scatter(*test_pos, c=test_labels)
ax[4].scatter(*test_pos, c=preds.mean.numpy())

ax[0].set_title("Original field")
ax[1].set_title("Normed field")
ax[2].set_title("Training data")
ax[3].set_title("Test data")
ax[4].set_title("Predictions")

[ax[i].set_aspect("equal") for i in range(5)]
