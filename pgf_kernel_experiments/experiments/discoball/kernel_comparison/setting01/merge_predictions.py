# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting01.set_env import num_runs, output_paths

# %% Merge and save predictions in a single file per run

kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

for run_count in range(num_runs):
    all_predictions = []

    for kernel_name in kernel_names:
        all_predictions.append(np.loadtxt(output_paths[run_count].joinpath(kernel_name+'_gp_predictions.csv')))

    all_predictions = np.column_stack(all_predictions)

    np.savetxt(
        output_paths[run_count].joinpath('predictions.csv'),
        all_predictions,
        delimiter=',',
        header=','.join(kernel_names),
        comments=''
    )
