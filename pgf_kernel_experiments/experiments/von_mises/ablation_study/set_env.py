# %% Import packages

from pathlib import Path
# %% Indicate whereas to use GPUs or CPUs

use_cuda = False # True

# %% Set paths

data_path = Path('data')
output_path = Path('output')

# %% Set seed

data_seed = 2000
train_seed = 102000
