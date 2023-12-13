# %% Import packages

import gpytorch
import numpy as np
import torch

from pgf_kernel_experiments.experiments.cox_process.feature_extractor import FeatureExtractor
from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, n, num_classes, num_runs, output_basepath, output_paths, use_cuda
)
from pgf_kernel_experiments.runners import ExactSingleDKLRunner

# %% Set maximum number of CG iterations

# gpytorch.settings.max_cg_iterations(300)

# %% Make and save predictions and error metrics per run

all_scores = []

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d} completed'

for run_count in range(num_runs):
    # Load data

    input_data = np.loadtxt(data_paths[run_count].joinpath('input_data.csv'), delimiter=',')

    # np.linalg.norm(input_data, axis=1)

    labels = np.loadtxt(data_paths[run_count].joinpath('labels.csv'), dtype='int')

    train_ids = np.loadtxt(data_paths[run_count].joinpath('train_ids.csv'), dtype='int')

    test_ids = np.loadtxt(data_paths[run_count].joinpath('test_ids.csv'), dtype='int')

    # Get training data

    train_pos = input_data[train_ids, :]
    train_output = labels[train_ids]

    # Get test data

    test_pos = input_data[test_ids, :]
    test_output = labels[test_ids]

    # Convert training data to PyTorch format

    train_x = torch.as_tensor(train_pos, dtype=torch.float64)
    train_y = torch.as_tensor(train_output.T, dtype=torch.int64)

    if use_cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()

    # Convert test data to PyTorch format

    test_x = torch.as_tensor(test_pos, dtype=torch.float64)
    test_y = torch.as_tensor(test_output.T, dtype=torch.int64)

    if use_cuda:
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    # Set up ExactSingleDKLRunner

    feature_extractor=FeatureExtractor(n+1)

    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(batch_shape=torch.Size((num_classes,))),
        batch_shape=torch.Size((num_classes,)),
    )

    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_y, learn_additional_noise=True)

    runner = ExactSingleDKLRunner(
        train_x,
        likelihood.transformed_targets,
        feature_extractor,
        kernel,
        likelihood,
        task='classification',
        num_classes=likelihood.num_classes,
        use_cuda=use_cuda
    )

    # Set the model in double mode

    runner.model.double()
    runner.model.likelihood.double()

    # Load model state

    runner.model.load_state_dict(torch.load(output_paths[run_count].joinpath('rbf_dkl_state.pth')))

    # Make predictions

    predictions = runner.test(test_x).max(0)[1]

    # Compute error metrics

    scores = runner.assess(
        predictions,
        test_y,
        metrics=[
            lambda predictions, y : (torch.sum(predictions == test_y) / len(test_y)).item(),
        ]
    )

    all_scores.append(scores)

    # Save predictions

    np.savetxt(
        output_paths[run_count].joinpath('rbf_dkl_predictions.csv'),
        predictions.cpu().detach().numpy(),
        fmt='%d'
    )

    # Save error metrics

    np.savetxt(
        output_paths[run_count].joinpath('rbf_dkl_error_metrics.csv'),
        [scores.cpu().detach().numpy()],
        delimiter=',',
        header='accuracy',
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
    output_basepath.joinpath('rbf_dkl_error_metric_means.csv'),
    [means.cpu().detach().numpy()],
    delimiter=',',
    header='accuracy',
    comments=''
)

np.savetxt(
    output_basepath.joinpath('rbf_dkl_error_metric_stds.csv'),
    [stds.cpu().detach().numpy()],
    delimiter=',',
    header='accuracy',
    comments=''
)
