# %% Import packages

import numpy as np
import torch

from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.von_mises.ablation_study.set_env import data_path, output_path, train_seed, use_cuda
from pgf_kernel_experiments.runners import ExactMultiGPRunner

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

torch.manual_seed(train_seed)

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

# %% Get training data

train_pos = grid[train_ids, :]
train_output = z[train_ids]

# %% Convert training data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output.T, dtype=torch.float64)

if use_cuda:
    train_x = train_x.cuda()
    train_y = train_y.cuda()

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[5]),
    GFKernel(width=[20]),
    GFKernel(width=[20, 5]),
    GFKernel(width=[20, 20]),
    GFKernel(width=[20, 20, 5]),
    GFKernel(width=[20, 20, 20]),
]

kernel_names = ['5', '20', '20_5', '20_20', '20_20_5', '20_20_20']

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels, use_cuda=use_cuda)

# %% Set the models in double mode

for i in range(runner.num_gps()):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

# %% Configure training setup for GP models

# list(runner.single_runners[0].model.named_parameters())

lrs = [0.8, 0.5, 2., 2., 2.]

optimizers = []

schedulers = []

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[0].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[0].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[0].model.covar_module.pars0, "lr": lrs[2]}
]))

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[1].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[1].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[1].model.covar_module.pars0, "lr": lrs[2]}
]))

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[2].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[2].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[2].model.covar_module.pars0, "lr": lrs[2]},
    {"params": runner.single_runners[2].model.covar_module.pars1, "lr": lrs[3]}
]))

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[3].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[3].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[3].model.covar_module.pars0, "lr": lrs[2]},
    {"params": runner.single_runners[3].model.covar_module.pars1, "lr": lrs[3]}
]))

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[4].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[4].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[4].model.covar_module.pars0, "lr": lrs[2]},
    {"params": runner.single_runners[4].model.covar_module.pars1, "lr": lrs[3]},
    {"params": runner.single_runners[4].model.covar_module.pars2, "lr": lrs[4]}
]))

optimizers.append(torch.optim.Adam([
    {"params": runner.single_runners[5].model.likelihood.noise_covar.raw_noise, "lr": lrs[0]},
    {"params": runner.single_runners[5].model.mean_module.raw_constant, "lr": lrs[1]},
    {"params": runner.single_runners[5].model.covar_module.pars0, "lr": lrs[2]},
    {"params": runner.single_runners[5].model.covar_module.pars1, "lr": lrs[3]},
    {"params": runner.single_runners[5].model.covar_module.pars2, "lr": lrs[4]}
]))

for i in range(runner.num_gps()):
    schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], T_0=20, T_mult=1, eta_min=0.05))

num_iters = 100

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
