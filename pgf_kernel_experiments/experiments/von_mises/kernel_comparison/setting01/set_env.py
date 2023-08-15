# %% Import packages

import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True
# use_cuda = False

# %% Set number of runs

# num_runs = 10
num_runs = 1

# %% Set seeds

data_seed = 1

init_train_seed = 100000
# num_train_seeds = 50 * num_runs
num_train_seeds = 5 * num_runs

train_seeds = torch.randint(0, 100*num_train_seeds, (num_train_seeds, ))

# %% Data simulation setup

num_samples = 1000

perc_training = 0.5

# %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]
