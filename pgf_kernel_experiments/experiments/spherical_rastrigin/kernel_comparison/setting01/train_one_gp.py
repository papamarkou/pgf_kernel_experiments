# %% Import packages

import gpytorch
import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.spherical_rastrigin.kernel_comparison.setting01.set_env import (
    data_path, output_path, seed
)
from pgf_kernel_experiments.runners import ExactMultiGPRunner

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

torch.manual_seed(seed+20)

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)

grid = data[:, 2:5]
x = data[:, 2]
y = data[:, 3]
z = data[:, 4]
v = data[:, 5]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')

# %% Get training data

train_pos = grid[train_ids, :]
train_output = v[train_ids]

# %% Convert training data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

if use_cuda:
    train_x = train_x.cuda()
    train_y = train_y.cuda()

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[30, 30, 30]),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    # gpytorch.kernels.PeriodicKernel(),
    # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=3)
]

kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Set the models in double mode

for i in range(runner.num_gps()):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

# %% Configure training setup for GP models

# list(runner.single_runners[0].model.named_parameters())

optimizers = []

schedulers = []

pgf_optim_per_group = True

num_iters = 5
# num_iters = 100
# num_iters = 500

if pgf_optim_per_group:
    # lrs = [[0.1, 0.1, 0.9, 0.9, 0.9], 0.05, 0.05, 0.05, 0.05]
    lrs = [[0.8, 0.5, 0.9, 0.9, 0.9], 0.1, 0.1, 0.1, 0.1]
    # lrs = [[0.9, 0.9, 0.9], 0.1, 0.1, 0.1, 0.1]

    optimizers.append(torch.optim.Adam([
        {"params": runner.single_runners[0].model.likelihood.noise_covar.raw_noise, "lr": lrs[0][0]},
        {"params": runner.single_runners[0].model.mean_module.raw_constant, "lr": lrs[0][1]},
        {"params": runner.single_runners[0].model.covar_module.pars0, "lr": lrs[0][2]},
        {"params": runner.single_runners[0].model.covar_module.pars1, "lr": lrs[0][3]},
        {"params": runner.single_runners[0].model.covar_module.pars2, "lr": lrs[0][4]}
    ]))

    # schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=num_iters, eta_min=0.1))
    schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], T_0=20, T_mult=1, eta_min=0.1))

    # optimizers.append(torch.optim.Adam([
    #     {"params": runner.single_runners[1].model.likelihood.noise_covar.raw_noise, "lr": 0.8},
    #     {"params": runner.single_runners[1].model.mean_module.raw_constant, "lr": 0.5},
    #     {"params": runner.single_runners[1].model.covar_module.raw_outputscale, "lr": 0.1},
    #     {"params": runner.single_runners[1].model.covar_module.base_kernel.raw_lengthscale, "lr": 0.1},
    # ]))

    # schedulers.append(None)

    # # for i in range(1, runner.num_gps()):
    # for i in range(2, runner.num_gps()):
    #     optimizers.append(torch.optim.Adam(
    #         runner.single_runners[i].model.parameters(), lr=lrs[i]
    #     ))

    #     # schedulers.append(
    #     #     torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max=num_iters, eta_min=0.01)
    #     # )
    #     schedulers.append(None)
else:
    lrs = [0.7, 0.5, 0.5, 0.075, 0.5]

    for i in range(runner.num_gps()):
        optimizers.append(torch.optim.Adam(runner.single_runners[i].model.parameters(), lr=lrs[i]))

        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[i], T_0=20, T_mult=1, eta_min=0.05)
        )

# %% Train GP models to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizers, num_iters, schedulers=schedulers)

# list(runner.single_runners[0].model.named_parameters())

# %% Save model states

for i in range(runner.num_gps()):
    torch.save(
        runner.single_runners[i].model.state_dict(),
        output_path.joinpath(kernel_names[i]+'_gp_state.pth')
    )

# %% Save losses

np.savetxt(
    output_path.joinpath('losses.csv'),
    losses.cpu().detach().numpy(),
    delimiter=',',
    header=','.join(kernel_names),
    comments=''
)
