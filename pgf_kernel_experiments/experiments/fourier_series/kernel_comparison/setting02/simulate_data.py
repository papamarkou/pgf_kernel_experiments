# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.fourier_series.fourier_series import fourier_series
from pgf_kernel_experiments.experiments.fourier_series.kernel_comparison.setting02.set_env import (
    data_paths, data_seed, num_runs
)

# %% Create paths if they don't exist

for i in range(num_runs):
    data_paths[i].mkdir(parents=True, exist_ok=True)

# %% Data simulation setup

num_samples = 1000

perc_train = 0.8

num_train_subset_samples = 100

# %% Simulate and save data

for i in range(num_runs):
    # Set seed

    np.random.seed(data_seed+i)

    # Generate all data

    theta = np.random.uniform(low=-np.pi, high=np.pi, size=num_samples)

    x = np.cos(theta)

    y = np.sin(theta)

    z_signal = fourier_series(
        theta,
        a=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        p=np.pi / 5,
        phi=np.array([0., 3 * np.pi, np.pi / 5, np.pi / 10])
    )

    z_noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=num_samples)

    z = z_signal + z_noise

    # Generate training data

    ids = np.arange(num_samples)

    num_train = int(perc_train * num_samples)

    train_ids = np.random.choice(ids, size=num_train, replace=False)

    train_ids.sort()

    # Generate training subsets

    train_subset_ids = np.random.choice(train_ids, size=num_train_subset_samples, replace=False)
    train_subset_ids.sort()

    # Generate test data

    test_ids = np.array(list(set(ids).difference(set(train_ids))))

    test_ids.sort()

    # Save data

    np.savetxt(
        data_paths[i].joinpath('data.csv'),
        np.column_stack([theta, x, y, z]),
        delimiter=',',
        header='theta,x,y,z',
        comments=''
    )

    np.savetxt(
        data_paths[i].joinpath('data.csv'),
        np.column_stack([theta, x, y, z_signal, z_noise, z]),
        delimiter=',',
        header='theta,x,y,z_signal,z_noise,z',
        comments=''
    )

    np.savetxt(data_paths[i].joinpath('train_ids.csv'), train_ids, fmt='%i')
    
    np.savetxt(data_paths[i].joinpath('train_subset_ids.csv'), train_subset_ids, fmt='%i')
    
    np.savetxt(data_paths[i].joinpath('test_ids.csv'), test_ids, fmt='%i')
