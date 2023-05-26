# %% Import packages

import numpy as np
import torch

from scipy.stats import vonmises

from pgf_kernel_experiments.experiments.von_mises.setting01.set_paths import data_path

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)

# %% Set seed

torch.manual_seed(1)

# %% Generate data

n_samples = 1000

theta = np.linspace(-np.pi, np.pi, num=n_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

z = vonmises.pdf(theta, kappa=2., loc=0., scale=0.05)

# %% Generate training data

ids = np.arange(n_samples)

n_train = int(0.5 * n_samples)

train_ids = np.random.choice(ids, size=n_train, replace=False)

train_ids.sort()

# %% Generate test data

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

n_test = n_samples - n_train

# %% Save data

np.savetxt(data_path.joinpath('theta.csv'), theta)

np.savetxt(data_path.joinpath('x.csv'), x, delimiter=',')
np.savetxt(data_path.joinpath('y.csv'), y, delimiter=',')
np.savetxt(data_path.joinpath('z.csv'), z, delimiter=',')

np.savetxt(data_path.joinpath('train_ids.csv'), train_ids, fmt='%i')
np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')
