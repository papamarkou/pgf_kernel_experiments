# %% Import packages

import gpytorch
import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting01.set_env import data_path, output_path
from pgf_kernel_experiments.runners import ExactMultiGPRunner

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

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

# %% Convert training data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

# %% Convert test data to PyTorch format

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output.T, dtype=torch.float64)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20, 20, 20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    gpytorch.kernels.PeriodicKernel(),
    gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
]

kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Set the models in double mode

for i in range(runner.num_gps()):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

# %% Load model states

for i in range(runner.num_gps()):
    runner.single_runners[i].model.load_state_dict(torch.load(output_path.joinpath(kernel_names[i]+'_gp_state.pth')))

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

# %% Save predictions

np.savetxt(
    output_path.joinpath('predictions.csv'),
    torch.stack([predictions[i].mean for i in range(runner.num_gps())], dim=0).t().detach().numpy(),
    delimiter=',',
    header=','.join(kernel_names),
    comments=''
)

# %% Save error metrics

np.savetxt(
    output_path.joinpath('error_metrics.csv'),
    scores.detach().numpy(),
    delimiter=',',
    header='mean_abs_error,mean_sq_error,loss',
    comments=''
)
