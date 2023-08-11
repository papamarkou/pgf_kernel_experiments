# %% Import packages

import gpytorch
import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting01.set_env import (
    data_paths, num_runs, output_basepath, output_paths, use_cuda
)
from pgf_kernel_experiments.runners import ExactMultiGPRunner

# %% Make and save predictions and error metrics per run

all_scores = []

kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d} completed'

for run_count in range(num_runs):
    # Load data

    data = np.loadtxt(
        data_paths[run_count].joinpath('data.csv'),
        delimiter=',',
        skiprows=1
    )

    grid = data[:, 1:3]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    train_ids = np.loadtxt(data_paths[run_count].joinpath('train_ids.csv'), dtype='int')

    test_ids = np.loadtxt(data_paths[run_count].joinpath('test_ids.csv'), dtype='int')

    # Get training data

    train_pos = grid[train_ids, :]
    train_output = z[train_ids]

    # Get test data

    test_pos = grid[test_ids, :]
    test_output = z[test_ids]

    # Convert training data to PyTorch format

    train_x = torch.as_tensor(train_pos, dtype=torch.float64)
    train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

    if use_cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()

    # Convert test data to PyTorch format

    test_x = torch.as_tensor(test_pos, dtype=torch.float64)
    test_y = torch.as_tensor(test_output.T, dtype=torch.float64)

    if use_cuda:
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    # Set up ExactMultiGPRunner

    kernels = [
        GFKernel(width=[30, 30, 30]),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
        gpytorch.kernels.PeriodicKernel(),
        gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
    ]

    runner = ExactMultiGPRunner.generator(train_x, train_y, kernels, use_cuda=use_cuda)

    # Set the models in double mode

    for i in range(runner.num_gps()):
        runner.single_runners[i].model.double()
        runner.single_runners[i].model.likelihood.double()

    # Load model states

    for i in range(runner.num_gps()):
        runner.single_runners[i].model.load_state_dict(
            torch.load(output_paths[run_count].joinpath(kernel_names[i]+'_gp_state.pth'))
        )

    # Make predictions

    predictions = runner.test(test_x)

    # Compute error metrics

    scores = runner.assess(
        predictions,
        test_y,
        metrics=[
            gpytorch.metrics.mean_absolute_error,
            gpytorch.metrics.mean_squared_error,
            lambda predictions, y : gpytorch.metrics.negative_log_predictive_density(predictions, y)
        ]
    )

    all_scores.append(scores)

    # Save predictions

    np.savetxt(
        output_paths[run_count].joinpath('predictions.csv'),
        torch.stack([predictions[i].mean for i in range(runner.num_gps())], dim=0).t().cpu().detach().numpy(),
        delimiter=',',
        header=','.join(kernel_names),
        comments=''
    )

    # Save error metrics

    np.savetxt(
        output_paths[run_count].joinpath('error_metrics.csv'),
        scores.cpu().detach().numpy(),
        delimiter=',',
        header='mean_abs_error,mean_sq_error,loss',
        comments=''
    )

    # If verbose, state run number

    if verbose:
        print(msg.format(run_count+1, num_runs))

# %% Compute error metric summaries across runs

all_scores = torch.stack(all_scores)

means = all_scores.mean(dim=0)

stds = all_scores.std(dim=0)

# %% Save error metric summaries across runs

np.savetxt(
    output_basepath.joinpath('error_metric_means.csv'),
    means.cpu().detach().numpy(),
    delimiter=',',
    header='mean_abs_error,mean_sq_error,loss',
    comments=''
)

np.savetxt(
    output_basepath.joinpath('error_metric_stds.csv'),
    stds.cpu().detach().numpy(),
    delimiter=',',
    header='mean_abs_error,mean_sq_error,loss',
    comments=''
)
