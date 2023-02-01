# %% Import packages

import gpytorch
import torch

import matplotlib.pyplot as plt
import numpy as np

from datetime import date
from dateutil import relativedelta

from pgf_kernel_experiments.runners import ExactSingleGPRunner
from pgfml.kernels import GFKernel

# %% Load data

dataset = np.loadtxt('passenger_numbers.csv')

n_samples = len(dataset)

# %% Plot data

fontsize = 11

plt.figure(figsize=(8, 4))

plt.plot(range(n_samples), dataset)

plt.ylim([100 - 35, 600 + 35])

plt.title('Air passenger data', fontsize=fontsize)

plt.xlabel('Date', fontsize=fontsize)
plt.ylabel('Number of passengers', fontsize=fontsize)

start_date = date.fromisoformat('1949-01-01')
month_counts = [i*12 for i in range(int(len(dataset) / 12) + 1)]
years = [(start_date + relativedelta.relativedelta(months=i)).year for i in month_counts]

plt.xticks(ticks=month_counts, labels=years, fontsize=fontsize, rotation=30)
plt.yticks(fontsize=fontsize)

# %% Place data on the unit circle

theta = np.linspace(-np.pi, np.pi, num=n_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

grid = np.stack((x, y))

# %% Generate training data

n_train = 96

train_ids = range(n_samples)[:n_train]

train_pos = grid[:, train_ids]

train_output = dataset[train_ids]

# %% Generate test data

test_ids = range(n_train, n_samples)

n_test = n_samples - n_train

test_pos = grid[:, test_ids]

test_output = dataset[test_ids]

# %% Plot training and test data

fontsize = 11

plt.figure(figsize=(8, 4))

plt.plot(range(n_train), train_output)

plt.plot(range(n_train, n_samples), test_output)

plt.ylim([100 - 35, 600 + 35])

plt.title('Air passenger data', fontsize=fontsize)

plt.xlabel('Date', fontsize=fontsize)
plt.ylabel('Number of passengers', fontsize=fontsize)

start_date = date.fromisoformat('1949-01-01')
month_counts = [i*12 for i in range(int(len(dataset) / 12) + 1)]
years = [(start_date + relativedelta.relativedelta(months=i)).year for i in month_counts]

plt.xticks(ticks=month_counts, labels=years, fontsize=fontsize, rotation=30)
plt.yticks(fontsize=fontsize)

plt.legend(['Training data', 'Test data'], fontsize=fontsize)

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos.T, dtype=torch.float32)
train_y = torch.as_tensor(train_output.T, dtype=torch.float32)

test_x = torch.as_tensor(test_pos.T, dtype=torch.float32)
test_y = torch.as_tensor(test_output.T, dtype=torch.float32)

# %% Set up ExactMultiGPRunner

runner = ExactSingleGPRunner(train_x, train_y, GFKernel(width=[20, 20, 20]))

# %% Configurate training setup for GP models

optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.5)

n_iters = 1000

# %% Train GP models to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizer, n_iters)

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

fontsize = 11

plt.figure(figsize=(8, 4))

plt.plot(range(n_train), train_output)

plt.plot(range(n_train, n_samples), test_output)

plt.plot(range(n_train, n_samples), predictions.mean.numpy())

plt.ylim([100 - 35, 600 + 35])

plt.title('Air passenger data', fontsize=fontsize)

plt.xlabel('Date', fontsize=fontsize)
plt.ylabel('Number of passengers', fontsize=fontsize)

start_date = date.fromisoformat('1949-01-01')
month_counts = [i*12 for i in range(int(len(dataset) / 12) + 1)]
years = [(start_date + relativedelta.relativedelta(months=i)).year for i in month_counts]

plt.xticks(ticks=month_counts, labels=years, rotation=30)
plt.yticks(fontsize=fontsize)

plt.legend(['Training data', 'Test data', 'PGF-GP predictions'], fontsize=fontsize)
