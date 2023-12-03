# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, data_seed, kappas, lambdas, n, num_runs, perc_train
)
from pgf_kernel_experiments.experiments.cox_process.cox_process import CoxProcess

# %% Create paths if they don't exist

for i in range(num_runs):
    data_paths[i].mkdir(parents=True, exist_ok=True)

# %% Simulate and save data

for i in range(num_runs):
    # Set seed

    np.random.seed(data_seed+i)

    # Set up Cox process

    cox_process = CoxProcess(n, lambdas, kappas)

    # Generate all data, with input data in Cartesian cordinates

    data_cartesian_coords, labels, center_cartesian_coords, num_points = cox_process.simulate_data()

    # np.linalg.norm(data_cartesian_coords, axis=1)

    num_samples = len(labels)

    # Generate training data

    ids = np.arange(num_samples)

    train_ids = np.random.choice(ids, size=int(perc_train*num_samples), replace=False)

    train_ids.sort()

    # Generate test data

    test_ids = np.array(list(set(ids).difference(set(train_ids))))

    np.random.shuffle(test_ids)

    # Save data

    np.savetxt(
        data_paths[i].joinpath('input_data.csv'),
        data_cartesian_coords,
        delimiter=',',
        header='',
        comments=''
    )

    np.savetxt(
        data_paths[i].joinpath('labels.csv'),
        labels,
        fmt='%d',
        header='',
        comments=''
    )

    np.savetxt(
        data_paths[i].joinpath('cluster_centers.csv'),
        center_cartesian_coords,
        delimiter=',',
        header='',
        comments=''
    )

    np.savetxt(
        data_paths[i].joinpath('num_points.csv'),
        num_points,
        fmt='%d',
        header='',
        comments=''
    )

    np.savetxt(data_paths[i].joinpath('train_ids.csv'), train_ids, fmt='%i')
    np.savetxt(data_paths[i].joinpath('test_ids.csv'), test_ids, fmt='%i')
