# %% Import packages

import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Set number of runs

num_runs = 10

# %% Training setup

num_train_iters = 100

# %% Set seeds

data_seed = 5000

num_gps = 5

init_train_seed = 405000
num_train_seeds = 5*num_gps*num_runs

train_seeds = torch.randint(init_train_seed, init_train_seed+100*num_train_seeds, (num_gps, num_train_seeds))

# %% Set scaling of trigonometric function

a = 6.

 # %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]
