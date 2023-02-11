# %% Import packages

import gpytorch
import torch

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.runners import ExactMultiGPRunner
from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.bloch_density.bloch_density import BlochDensity
from pgf_kernel_experiments.experiments.bloch_density.set_paths import data_path

# %% Load data

phi = np.loadtxt(data_path.joinpath('phi.csv'))
theta = np.loadtxt(data_path.joinpath('theta.csv'))

x = np.loadtxt(data_path.joinpath('x.csv'), delimiter=',')[:-1, :]
y = np.loadtxt(data_path.joinpath('y.csv'), delimiter=',')[:-1, :]
z = np.loadtxt(data_path.joinpath('z.csv'), delimiter=',')[:-1, :]

freqs = np.loadtxt(data_path.joinpath('freqs.csv'), delimiter=',')[:-1, :]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

freqs_flat = freqs.flatten()

pos = np.column_stack((x_flat, y_flat, z_flat))

# %% Set up training and test data

train_pos = pos[train_ids, :]

train_output = freqs_flat[train_ids]

test_pos = pos[test_ids, :]

test_output = freqs_flat[test_ids]

# %% Generate BlochDensity for all data

n_train_freqs = int((freqs.shape[0] - 1) / 2)

phi_front = phi[:n_train_freqs]
theta_front = theta
x_front = x[:n_train_freqs, :]
y_front = y[:n_train_freqs, :]
z_front = z[:n_train_freqs, :]
freqs_front = freqs[:n_train_freqs, :]

phi_back = phi[(n_train_freqs - 1):]
theta_back = theta
x_back = x[(n_train_freqs - 1):, :]
y_back = y[(n_train_freqs - 1):, :]
z_back = z[(n_train_freqs - 1):, :]
freqs_back = freqs[(n_train_freqs - 1):, :]

bloch_all_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, freqs_front,
    phi_back, theta_back, x_back, y_back, z_back, freqs_back,
    alpha = 0.33
)

# %% Generate BlochDensity for training data

train_freqs_plot = freqs.copy()
train_freqs_plot = train_freqs_plot.flatten()
train_freqs_plot[test_ids] = np.nan
train_freqs_plot = train_freqs_plot.reshape(*(freqs.shape))

train_freqs_plot_front = train_freqs_plot[:n_train_freqs, :]

train_freqs_plot_back = train_freqs_plot[(n_train_freqs - 1):, :]

bloch_train_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, train_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, train_freqs_plot_back,
    alpha = 0.33
)

# %% Generate BlochDensity for test data

test_freqs_plot = freqs.copy()
test_freqs_plot = test_freqs_plot.flatten()
test_freqs_plot[train_ids] = np.nan
test_freqs_plot = test_freqs_plot.reshape(*(freqs.shape))

test_freqs_plot_front = test_freqs_plot[:n_train_freqs, :]

test_freqs_plot_back = test_freqs_plot[(n_train_freqs - 1):, :]

bloch_test_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, test_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, test_freqs_plot_back,
    alpha = 0.33
)

# %% Plot data

# https://qutip.org/docs/4.0.2/guide/guide-bloch.html

fontsize = 18

fig = plt.figure(figsize=[16, 6], constrained_layout=True)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

bloch_all_data.fig = fig
bloch_all_data.axes = ax1

bloch_all_data.xlpos = [1.55, -1.1]
bloch_all_data.zlpos = [1.22, -1.35]

bloch_all_data.render()

ax1.set_box_aspect([1, 1, 1]) 

ax1.set_title('All data', fontsize=fontsize)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')

bloch_train_data.fig = fig
bloch_train_data.axes = ax2

bloch_train_data.xlpos = [1.55, -1.1]
bloch_train_data.zlpos = [1.22, -1.35]

bloch_train_data.render()

ax2.set_box_aspect([1, 1, 1])

ax2.set_title('Training data', fontsize=fontsize)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')

bloch_test_data.fig = fig
bloch_test_data.axes = ax3

bloch_test_data.xlpos = [1.55, -1.1]
bloch_test_data.zlpos = [1.22, -1.35]

bloch_test_data.render()

ax3.set_box_aspect([1, 1, 1])

ax3.set_title('Test data', fontsize=fontsize)

# plt.show()

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output, dtype=torch.float64)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20, 20, 20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    gpytorch.kernels.PeriodicKernel(),
    gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=3)
]

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Set the models in double mode

for i in range(len(kernels)):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

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

# %%
