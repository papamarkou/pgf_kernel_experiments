# %% Import packages

import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Set number of runs

num_runs = 10

# %% Set seeds

data_seed = 4000

init_train_seed = 104000
num_train_seeds = 5*num_runs

train_seeds = torch.randint(init_train_seed, init_train_seed+100*num_train_seeds, (num_train_seeds, ))

init_train_subset_seed = 203000
num_train_subset_seeds = 5*num_runs

train_subset_seeds = torch.randint(
    init_train_subset_seed, init_train_subset_seed+100*num_train_subset_seeds, (num_train_subset_seeds, )
)

# %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]
