# %% Import packages

import numpy as np
import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Set number of runs

num_runs = 2 # 4

# %% Data simulation setup

perc_train = 0.5
# num_test = None

# %% Training setup

num_train_iters = 1000

# %% Set seeds

data_seed = 7000

num_gps = 5

init_train_seed = 607000
num_train_seeds = 5*num_gps*num_runs

train_seeds = torch.randint(init_train_seed, init_train_seed+100*num_train_seeds, (num_gps, num_train_seeds))

# %% Set hyperparameters of Cox process

n = 19
num_clusters = 10
lambdas = np.full(num_clusters, 850.)
kappas = np.full(num_clusters, 20.)

 # %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

# %% Set DPI for saving plots

dpi = 600
