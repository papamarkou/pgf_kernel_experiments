# %% Import packages

import torch

from pathlib import Path

# %% Indicate whereas to use GPUs or CPUs

use_cuda = True

# %% Set number of runs

num_runs = 10

# %% Set seeds

data_seed = 1

init_train_seed = 100000
num_train_seeds = 5 * num_runs

train_seeds = torch.randint(0, 100*num_train_seeds, (num_train_seeds, ))

# %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_basepath = Path('output')

output_paths = [Path(output_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]
