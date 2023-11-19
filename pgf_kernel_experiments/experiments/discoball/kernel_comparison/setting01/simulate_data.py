# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.discoball.kernel_comparison.setting01.set_env import (
    a, data_paths, data_seed, num_incl, num_runs, num_train
)
from pgf_kernel_experiments.experiments.discoball.discoball import gen_discoball_data

# %% Create paths if they don't exist

for i in range(num_runs):
    data_paths[i].mkdir(parents=True, exist_ok=True)

# %% Simulate and save data

for i in range(num_runs):
    # Set seed

    np.random.seed(data_seed+i)

    # Generate azimuth phi for polar coordinates of all data

    phi = np.random.uniform(low=-np.pi, high=np.pi, size=2*num_incl)
    phi.sort()

    # Generate inclination theta for polar coordinates of all data

    theta = np.tile(np.random.uniform(low=0, high=np.pi, size=num_incl), 2)
    theta.sort()

    # Generate all data, with input data in Cartesian cordinates

    x, y, z, v = gen_discoball_data(phi, theta, a=a)

    num_samples = np.size(v)

    # Generate training data

    ids = np.arange(num_samples)

    train_ids = np.random.choice(ids, size=num_train, replace=False)

    train_ids.sort()

    # Generate test data

    test_ids = np.array(list(set(ids).difference(set(train_ids))))

    np.random.shuffle(test_ids)

    # Save data

    np.savetxt(
        data_paths[i].joinpath('data.csv'),
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

    np.savetxt(data_paths[i].joinpath('dims.csv'), np.array([np.size(phi), np.size(theta)]), fmt='%i')

    np.savetxt(data_paths[i].joinpath('train_ids.csv'), train_ids, fmt='%i')
    np.savetxt(data_paths[i].joinpath('test_ids.csv'), test_ids, fmt='%i')
