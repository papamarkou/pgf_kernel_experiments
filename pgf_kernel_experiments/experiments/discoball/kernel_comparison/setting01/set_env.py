# %% Import packages

import numpy as np
import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Set number of runs

num_runs = 1
# num_runs = 4

# %% Data simulation setup

num_incl = 200

num_train = 4000
num_test = 4000
# num_test = None

# %% Training setup

# num_train_iters = 100
num_train_iters = 1000
# num_train_iters = 3000

# %% Set seeds

data_seed = 5000

num_gps = 5

init_train_seed = 405000
num_train_seeds = 5*num_gps*num_runs

train_seeds = torch.randint(init_train_seed, init_train_seed+100*num_train_seeds, (num_gps, num_train_seeds))

# %% Set scaling of discoball function

k = [0.5, 0.5, 0.5, 0.5]
l = [15, 15, 15, 15]
m = [0., np.pi / 2, np.pi, 3 * np.pi / 2]
terms = [0, 2, 1, 2]
a = 1.
b = 0.

 # %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

# %% Set DPI for saving plots

dpi = 600
