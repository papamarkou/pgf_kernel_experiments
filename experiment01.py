# %% Import packages

import gpytorch
import math
import torch

from matplotlib import pyplot as plt

from pgfml.kernels import GFKernel

# %% Generate training data

# Training data: 200 points in [-1, 1], inclusive, regularly spaced
train_x = torch.linspace(-1., 1., 200)

# Objective function: sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

# %% Plot training data

plt.scatter(train_x, train_y)

# %% Define GP model for exact inference

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(GFKernel(num_p=[100]))
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

num_training_iters = 1000

# Get to training mode
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

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
    # Test points are regularly spaced along [-1, 1]
    test_x = torch.linspace(-1., 1., 51)

    # Make predictions by feeding model through likelihood
    observed_pred = likelihood(model(test_x))

# %% Plot the fitted model

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # Configuration of plot
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
