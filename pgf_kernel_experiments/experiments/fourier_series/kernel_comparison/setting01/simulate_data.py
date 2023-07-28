# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.fourier_series.fourier_series import fourier_series
from pgf_kernel_experiments.experiments.fourier_series.kernel_comparison.setting01.set_env import data_path, seed

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

np.random.seed(seed)

# %% Generate data

num_samples = 500
# num_samples = 1000

theta = np.linspace(-np.pi, np.pi, num=num_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

z = fourier_series(
    theta,
    a=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    p=np.pi / 5,
    phi=np.array([0., 3 * np.pi, np.pi / 5, np.pi / 10])
)

# %% Generate training data

ids = np.arange(num_samples)

num_train = int(0.2 * num_samples)

train_ids = np.random.choice(ids, size=num_train, replace=False)

train_ids.sort()

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

# %% Save data

np.savetxt(
    data_path.joinpath('data.csv'),
    np.column_stack([theta, x, y, z]),
    delimiter=',',
    header='theta,x,y,z',
    comments=''
)

np.savetxt(data_path.joinpath('train_ids.csv'), train_ids, fmt='%i')
np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')
