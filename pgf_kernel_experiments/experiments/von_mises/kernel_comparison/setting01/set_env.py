# %% Import packages

from pathlib import Path

# %% Set seed

data_seed = 1
train_seed = 100000

# %% Set number of runs

num_runs = 10

# %% Data simulation setup

num_samples = 1000

perc_training = 0.5

# %% Set paths

data_basepath = Path('data')

data_paths = [Path(data_basepath, 'run'+str(i+1).zfill(len(str(num_runs)))) for i in range(num_runs)]

output_path = Path('output')
