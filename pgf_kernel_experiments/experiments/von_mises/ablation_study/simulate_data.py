# %% Import packages

import numpy as np

from scipy.stats import vonmises

from pgf_kernel_experiments.experiments.von_mises.ablation_study.set_env import data_path, data_seed

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)

# %% Data simulation setup

num_samples = 500

perc_train = 0.5

# %% Set seeds

np.random.seed(data_seed)

# %% Generate data

theta = np.random.default_rng().uniform(low=-np.pi, high=np.pi, size=num_samples)
theta.sort()

x = np.cos(theta)

y = np.sin(theta)

z_signal = vonmises.pdf(theta, kappa=2., loc=0., scale=0.05)

z_noise = np.random.default_rng().normal(loc=0.0, scale=0.5, size=num_samples)

z = z_signal + z_noise

# %% Generate training data

ids = np.arange(num_samples)

num_train = int(perc_train * num_samples)

train_ids = np.random.choice(ids, size=num_train, replace=False)

train_ids.sort()

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
np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')
