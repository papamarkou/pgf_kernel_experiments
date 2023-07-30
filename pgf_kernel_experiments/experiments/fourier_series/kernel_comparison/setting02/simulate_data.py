# %% Import packages

import numpy as np

from pgf_kernel_experiments.experiments.fourier_series.fourier_series import fourier_series
from pgf_kernel_experiments.experiments.fourier_series.kernel_comparison.setting02.set_env import data_path, seed

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

np.random.seed(seed)

# %% Generate data

num_samples = 1000

theta = np.linspace(-np.pi, np.pi, num=num_samples, endpoint=False)

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

# %% Generate training data

ids = np.arange(num_samples)

num_train = int(800)

train_ids = np.random.choice(ids, size=num_train, replace=False)

train_ids.sort()

# %% Generate training subsets

train_subset1_ids = np.random.choice(train_ids, size=600, replace=False)
train_subset1_ids.sort()

train_subset2_ids = np.random.choice(train_subset1_ids, size=400, replace=False)
train_subset2_ids.sort()

train_subset3_ids = np.random.choice(train_subset2_ids, size=200, replace=False)
train_subset3_ids.sort()

train_subset4_ids = np.random.choice(train_subset3_ids, size=100, replace=False)
train_subset4_ids.sort()

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

# %% Save data

np.savetxt(
    data_path.joinpath('data.csv'),
    np.column_stack([theta, x, y, z_signal, z_noise, z]),
    delimiter=',',
    header='theta,x,y,z_signal,z_noise,z',
    comments=''
)

np.savetxt(data_path.joinpath('train_ids.csv'), train_ids, fmt='%i')

np.savetxt(data_path.joinpath('train_subset1_ids.csv'), train_subset1_ids, fmt='%i')
np.savetxt(data_path.joinpath('train_subset2_ids.csv'), train_subset2_ids, fmt='%i')
np.savetxt(data_path.joinpath('train_subset3_ids.csv'), train_subset3_ids, fmt='%i')
np.savetxt(data_path.joinpath('train_subset4_ids.csv'), train_subset4_ids, fmt='%i')

np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')
