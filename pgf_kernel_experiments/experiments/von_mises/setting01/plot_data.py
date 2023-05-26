# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.von_mises.setting01.set_paths import data_path

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)
