# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting01.set_env import (
    data_paths, data_seed, num_runs
)
from pgf_kernel_experiments.experiments.trigonometric.trigonometric import gen_trigonometric_data

# %% Create paths if they don't exist

for i in range(num_runs):
    data_paths[i].mkdir(parents=True, exist_ok=True)

# %% Data simulation setup

num_incl = 200

num_train = 4000
num_test = 4000

# %% Simulate and save data

for i in range(num_runs):
    # Set seed

    np.random.seed(data_seed+i)

    # Azimuth phi
    phi = np.random.uniform(low=-np.pi, high=np.pi, size=2*num_incl)
    phi.sort()

    # Inclination theta
    theta = np.tile(np.random.uniform(low=0, high=np.pi, size=num_incl), 2)
    theta.sort()

    x, y, z, v = gen_trigonometric_data(phi, theta)

    num_samples = np.size(v)

    # Generate training data

    ids = np.arange(num_samples)

    train_ids = np.random.choice(ids, size=num_train, replace=False)

    train_ids.sort()

    # Generate test data

    test_ids = np.array(list(set(ids).difference(set(train_ids))))
    test_ids = np.random.choice(test_ids, size=num_test, replace=False)

    test_ids.sort()

    # Save data

    np.savetxt(
        data_path.joinpath('data.csv'),
        np.column_stack([
            np.outer(phi, np.ones(np.size(theta))).flatten(),
            np.outer(np.ones(np.size(phi)), theta).flatten(),
            x.flatten(),
            y.flatten(),
            z.flatten(),
            v.flatten()
        ]),
        delimiter=',',
        header='phi,theta,x,y,z,v',
        comments=''
    )

    np.savetxt(data_path.joinpath('dims.csv'), np.array([np.size(phi), np.size(theta)]), fmt='%i')

    np.savetxt(data_path.joinpath('train_ids.csv'), train_ids, fmt='%i')
    np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')