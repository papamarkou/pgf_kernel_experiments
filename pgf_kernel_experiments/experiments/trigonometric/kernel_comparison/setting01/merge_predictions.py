# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting01.set_env import num_runs, output_paths

# %% Load predictions based on different kernels

kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

# %% Merge and save predictions into a single file
