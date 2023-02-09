# %% Import packages

import gpytorch
import math
import torch

from matplotlib import pyplot as plt

from pgf_kernel_experiments.runners import ExactSingleGPRunner
from pgfml.kernels import GFKernel

# %% Generate training data

# Training data: 200 points in [-1, 1], inclusive, regularly spaced
train_x = torch.linspace(-1., 1., 200, dtype=torch.float64)

# Objective function: sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

# %% Generate test data

test_x = torch.linspace(-1., 1., 50)

# %% Convert training and test data from float32 to float64

train_x = train_x.to(torch.float64)
train_y = train_y.to(torch.float64)

test_x = test_x.to(torch.float64)

# %% Plot training data

plt.figure(figsize=(8, 4))

plt.scatter(train_x, train_y)

# %% Set up ExactMultiGPRunner

runner = ExactSingleGPRunner(train_x, train_y, gpytorch.kernels.ScaleKernel(GFKernel(width=[100])))

# %% Set the model in double mode

runner.model.double()
runner.model.likelihood.double()

# %% Print parameter names and values

for param_name, param_value in runner.model.named_parameters():
    print('Parameter name: {}. Parameter value: {}'.format(param_name, param_value))

# %% Configurate training setup for GP models

optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.1)

n_iters = 1000

# %% Train GP models to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizer, n_iters)

# %% Make predictions

predictions = runner.test(test_x)

# %% Plot the fitted model

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Get upper and lower confidence bounds
    lower, upper = predictions.confidence_region()

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), predictions.mean.numpy(), 'b')

    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # Configuration of plot
    ax.set_ylim([-3, 3])
    ax.legend(['Observed data', 'Mean', 'Confidence'])
