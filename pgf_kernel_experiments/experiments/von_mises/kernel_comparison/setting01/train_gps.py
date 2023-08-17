# %% Import packages

import gpytorch
import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting01.set_env import (
    data_paths, num_runs, num_train_iters, num_train_seeds, output_basepath, output_paths, train_seeds, use_cuda
)
from pgf_kernel_experiments.runners import ExactMultiGPRunner

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

    torch.manual_seed(train_seeds[tot_count])

    try:
        # Load data

        data = np.loadtxt(
            data_paths[success_count].joinpath('data.csv'),
            delimiter=',',
            skiprows=1
        )

        grid = data[:, 1:3]
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]

        train_ids = np.loadtxt(data_paths[success_count].joinpath('train_ids.csv'), dtype='int')

        # Get training data

        train_pos = grid[train_ids, :]
        train_output = z[train_ids]

        # Convert training data to PyTorch format

        train_x = torch.as_tensor(train_pos, dtype=torch.float64)
        train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        # Set ExactMultiGPRunner

        kernels = [
            GFKernel(width=[30, 30, 30]),
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
            gpytorch.kernels.PeriodicKernel(),
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
        ]

        kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

        runner = ExactMultiGPRunner.generator(train_x, train_y, kernels, use_cuda=use_cuda)

        # Set the models in double mode

        for i in range(runner.num_gps()):
            runner.single_runners[i].model.double()
            runner.single_runners[i].model.likelihood.double()

        # Configure training setup for GP models

        # for i in range(runner.num_gps()):
        #     print("Parameter names of runner {}".format(i))
        #     print(list(runner.single_runners[i].model.named_parameters()))

        ## Set optimizers

        optimizers = []

        ### Set optimizer for GFKernel

        optimizers.append(torch.optim.Adam([
            {"params": runner.single_runners[0].model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.single_runners[0].model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.single_runners[0].model.covar_module.pars0, "lr": 5.5},
            {"params": runner.single_runners[0].model.covar_module.pars1, "lr": 5.5},
            {"params": runner.single_runners[0].model.covar_module.pars2, "lr": 5.5}
        ]))

        ### Set optimizer for RBFKernel

        optimizers.append(torch.optim.Adam([
            {"params": runner.single_runners[1].model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.single_runners[1].model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.single_runners[1].model.covar_module.raw_outputscale, "lr": 0.5},
            {"params": runner.single_runners[1].model.covar_module.base_kernel.raw_lengthscale, "lr": 0.5}
        ]))

        ### Set optimizer for MaternKernel

        optimizers.append(torch.optim.Adam([
            {"params": runner.single_runners[2].model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.single_runners[2].model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.single_runners[2].model.covar_module.raw_outputscale, "lr": 0.5},
            {"params": runner.single_runners[2].model.covar_module.base_kernel.raw_lengthscale, "lr": 0.5}
        ]))

        ### Set optimizer for PeriodicKernel

        optimizers.append(torch.optim.Adam([
            {"params": runner.single_runners[3].model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.single_runners[3].model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.single_runners[3].model.covar_module.raw_lengthscale, "lr": 0.075},
            {"params": runner.single_runners[3].model.covar_module.raw_period_length, "lr": 0.075}
        ]))

        ### Set optimizer for SpectralMixtureKernel

        optimizers.append(torch.optim.Adam([
            {"params": runner.single_runners[4].model.likelihood.noise_covar.raw_noise, "lr": 0.1},
            {"params": runner.single_runners[4].model.mean_module.raw_constant, "lr": 0.1},
            {"params": runner.single_runners[4].model.covar_module.raw_mixture_weights, "lr": 0.5},
            {"params": runner.single_runners[4].model.covar_module.raw_mixture_means, "lr": 0.5},
            {"params": runner.single_runners[4].model.covar_module.raw_mixture_scales, "lr": 0.5}
        ]))

        ## Set schedulers

        schedulers = []

        ### Set scheduler for GFKernel

        schedulers.append(
        #     # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], T_0=50, T_mult=1, eta_min=2.0)
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[0],
                base_lr=[0.05, 0.05, 2, 2, 2],
                max_lr=[0.1, 0.1, 5.5, 5.5, 5.5],
                step_size_up=25,
                mode='triangular',
                cycle_momentum=False
            )
            # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=num_train_iters, eta_min=2.0)
            # torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[400, 470], gamma=0.5)
        )

        ### Set scheduler for RBFKernel

        schedulers.append(
            # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[i], T_0=50, T_mult=1, eta_min=0.05)
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[1],
                base_lr=[0.05, 0.05, 0.05, 0.05],
                max_lr=[0.1, 0.1, 0.5, 0.5],
                step_size_up=25,
                mode='triangular',
                cycle_momentum=False
            )
            # None
        )

        ### Set scheduler for MaternKernel

        schedulers.append(
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[2],
                base_lr=[0.05, 0.05, 0.05, 0.05],
                max_lr=[0.1, 0.1, 0.5, 0.5],
                step_size_up=25,
                mode='triangular',
                cycle_momentum=False
            )
        )

        ### Set scheduler for PeriodicKernel

        schedulers.append(
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[3],
                base_lr=[0.05, 0.05, 0.05, 0.05],
                max_lr=[0.1, 0.1, 0.075, 0.075],
                step_size_up=25,
                mode='triangular',
                cycle_momentum=False
            )
        )

        ### Set scheduler for SpectralMixtureKernel

        schedulers.append(
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[4],
                base_lr=[0.05, 0.05, 0.05, 0.05, 0.05],
                max_lr=[0.1, 0.1, 0.5, 0.5, 0.5],
                step_size_up=25,
                mode='triangular',
                cycle_momentum=False
            )
        )

        # Train GP models to find optimal hyperparameters

        losses = runner.train(train_x, train_y, optimizers, num_train_iters, schedulers=schedulers)

        # Save model states

        for i in range(runner.num_gps()):
            torch.save(
                runner.single_runners[i].model.state_dict(),
                output_paths[success_count].joinpath(kernel_names[i]+'_gp_state.pth')
            )

        # Save losses

        np.savetxt(
            output_paths[success_count].joinpath('losses.csv'),
            losses.cpu().detach().numpy(),
            delimiter=',',
            header=','.join(kernel_names),
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
    output_basepath.joinpath('successful_seeds.csv'),
    np.array(successful_seeds),
    fmt='%i'
)

np.savetxt(
    output_basepath.joinpath('failed_seeds.csv'),
    np.array(failed_seeds),
    fmt='%i'
)
