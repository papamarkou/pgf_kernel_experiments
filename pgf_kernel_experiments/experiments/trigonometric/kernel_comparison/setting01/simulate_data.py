# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.trigonometric.kernel_comparison.setting01.set_env import data_path, seed
from pgf_kernel_experiments.experiments.trigonometric.trigonometric import gen_trigonometric_data

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

np.random.seed(seed)

# %% Generate data

# num_incl = 100
num_incl = 200

# Azimuth phi
phi = np.random.uniform(low=-np.pi, high=np.pi, size=2*num_incl)
phi.sort()

# Inclination theta
theta = np.tile(np.random.uniform(low=0, high=np.pi, size=num_incl), 2)
theta.sort()

a = 30.
b = [0.0025, 0.002]
c = 10
# a = 30.
# b = [0.0202, 0.0205]
# c = 10

x, y, z, v = gen_trigonometric_data(phi, theta, a, b, c)

num_samples = np.size(v)

# %% Generate training data

ids = np.arange(num_samples)

# num_train = int(0.5 * num_samples)
# num_train = int(0.015 * num_samples)
# num_train = 2500

train_ids = np.random.choice(ids, size=4000, replace=False)

train_ids.sort()

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))
# test_ids = np.random.choice(test_ids, size=10000, replace=False)
test_ids = np.random.choice(test_ids, size=4000, replace=False)

test_ids.sort()

# %% Save data

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
