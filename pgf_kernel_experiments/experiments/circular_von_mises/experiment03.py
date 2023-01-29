# %% Import packages

import gpytorch
import torch

import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import vonmises

from pgfml.kernels import GFKernel

# %% Simulate data

n_samples = 3600

theta = np.linspace(-np.pi, np.pi, num=n_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

grid = np.stack((x, y))

z = vonmises.pdf(theta, kappa=2., loc=0., scale=0.05)

# %% Generate training data

ids = np.arange(n_samples)

n_train = int(0.75 * n_samples)

train_samples = np.random.RandomState(300).choice(ids, size=n_train, replace=False)

train_pos = grid[:, train_samples]

train_labels = z[train_samples]

# %% Generate test data

test_samples = np.array(list(set(ids).difference(set(train_samples))))

n_test = n_samples - n_train

test_pos = grid[:, test_samples]

test_labels = z[test_samples]

# %% Plot training and test data

fontsize = 10

titles = ['von Mises', 'Training data', 'Test data']

cols = ['green', 'orange', 'brown']

fig, ax = plt.subplots(1, 3, figsize=[12, 3], subplot_kw={'projection': '3d'})

ax[0].plot(x, y, z, color=cols[0], lw=2)

ax[1].scatter(train_pos[0, :], train_pos[1, :], train_labels, color=cols[1], s=2)

ax[2].scatter(test_pos[0, :], test_pos[1, :], test_labels, color=cols[2], s=2)

for i in range(3):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color='black', lw=2, zorder=0)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 0.6))

    ax[i].set_xticks([-1, 0, 1])
    ax[i].set_yticks([-1, 0, 1])
    ax[i].set_zticks([0, 0.3, 11.])

    ax[i].set_xlabel('x', fontsize=fontsize)
    ax[i].set_ylabel('y', fontsize=fontsize)
    ax[i].set_zlabel('z', fontsize=fontsize)

    ax[i].set_title(titles[i], fontsize=fontsize)

    ax[i].zaxis.set_rotate_label(False)

    ax[i].grid(False)

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
        # self.covar_module = GFKernel(width=[100])
        self.covar_module = gpytorch.kernels.ScaleKernel(GFKernel(width=[100]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

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

fontsize = 10

titles = ['von Mises', 'Training data', 'Test data', 'Predictions']

cols = ['green', 'orange', 'brown', 'red']

fig, ax = plt.subplots(1, 4, figsize=[16, 3], subplot_kw={'projection': '3d'})

ax[0].plot(x, y, z, color=cols[0], lw=2)

ax[1].scatter(train_pos[0, :], train_pos[1, :], train_labels, color=cols[1], s=2)

ax[2].scatter(test_pos[0, :], test_pos[1, :], test_labels, color=cols[2], s=2)

ax[3].scatter(test_pos[0, :], test_pos[1, :], preds.mean, color=cols[3], s=2)

for i in range(4):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color='black', lw=2, zorder=0)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 0.6))

    ax[i].set_xticks([-1, 0, 1])
    ax[i].set_yticks([-1, 0, 1])
    ax[i].set_zticks([0, 0.3, 11.])

    ax[i].set_xlabel('x', fontsize=fontsize)
    ax[i].set_ylabel('y', fontsize=fontsize)
    ax[i].set_zlabel('z', fontsize=fontsize)

    ax[i].set_title(titles[i], fontsize=fontsize)

    ax[i].zaxis.set_rotate_label(False)

    ax[i].grid(False)

# %%
