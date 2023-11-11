# %% Import packages

import gpytorch
import numpy as np
import torch

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting02.set_env import (
    data_paths, num_runs, num_train_iters, num_train_seeds, output_basepath, output_paths, train_seeds, use_cuda
)
from pgf_kernel_experiments.runners import ExactSingleGPRunner

# %% Create paths if they don't exist

for i in range(num_runs):
    output_paths[i].mkdir(parents=True, exist_ok=True)

# %% Run training and save model states

success_count = 0
tot_count = 0

successful_seeds = []
failed_seeds = []

verbose = True
if verbose:
    num_train_seed_digits = len(str(num_train_seeds))
    msg = 'Run {:'+str(num_train_seed_digits)+'d} {}'

while ((success_count < num_runs) and (tot_count < num_train_seeds)):
    # Set seed

    torch.manual_seed(train_seeds[2, tot_count])

    try:
        # Load data

        data = np.loadtxt(
            data_paths[success_count].joinpath('data.csv'),
            delimiter=',',
            skiprows=1
        )

        grid = data[:, 2:5]
        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]
        v = data[:, 5]

        train_ids = np.loadtxt(data_paths[success_count].joinpath('train_ids.csv'), dtype='int')

        # Get training data
        
        train_pos = grid[train_ids, :]
        train_output = v[train_ids]

        # Convert training data to PyTorch format

        train_x = torch.as_tensor(train_pos, dtype=torch.float64)
        train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        # Set # %% Set up ExactSingleGPRunner

        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

        runner = ExactSingleGPRunner(train_x, train_y, kernel, use_cuda=use_cuda)

        # Set the model in double mode

        runner.model.double()
        runner.model.likelihood.double()

        # Set optimizer

        optimizer =torch.optim.Adam([
            {"params": runner.model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.model.covar_module.raw_outputscale, "lr": 0.1},
            {"params": runner.model.covar_module.base_kernel.raw_lengthscale, "lr": 0.1}
        ])

        # Set scheduler

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=[0.01, 0.01, 0.01, 0.01],
            max_lr=[0.1, 0.1, 0.1, 0.1],
            step_size_up=25,
            scale_fn=lambda x : 0.97 ** (x - 1),
            cycle_momentum=False
        )

        # Train GP model to find optimal hyperparameters

        losses = runner.train(train_x, train_y, optimizer, num_train_iters, scheduler=scheduler)

        # Save model state

        torch.save(runner.model.state_dict(), output_paths[success_count].joinpath('matern_gp_state.pth'))

        # Save losses

        np.savetxt(
            output_paths[success_count].joinpath('matern_gp_losses.csv'),
            losses.cpu().detach().numpy(),
            delimiter=',',
            comments=''
        )

        # Housekeeping updates

        successful_seeds.append(train_seeds[tot_count])

        success_count = success_count + 1

        if verbose:
            success = True
    except:
        # Housekeeping updates

        failed_seeds.append(train_seeds[tot_count])

        if verbose:
            success = False

    tot_count = tot_count + 1

    if verbose:
        print(msg.format(tot_count, 'succeeded' if success else 'failed'))

# %% Save successful and failed seeds

np.savetxt(
    output_basepath.joinpath('matern_gp_successful_seeds.csv'),
    np.array(successful_seeds),
    fmt='%i'
)

np.savetxt(
    output_basepath.joinpath('matern_gp_failed_seeds.csv'),
    np.array(failed_seeds),
    fmt='%i'
)
