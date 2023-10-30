# %% Import packages

import numpy as np
import scipy

from pgf_kernel_experiments.experiments.rastrigin.set_paths import data_path

# %% Set seed

np.random.seed(9)

# %% Function for computing spherical Rastrigin function at a single point

def rastrigin_function(phi, theta, a, b):
    result = 2 * a
    # result = result + ((phi / b[0]) ** 2) - a * np.cos(2 * np.pi * phi / b[0])
    # result = result + ((theta / b[1]) ** 2) - a * np.cos(2 * np.pi * theta / b[1])
    result = result + (phi ** 2) - a * np.cos(2 * np.pi * phi / b[0])
    result = result + (theta ** 2) - a * np.cos(2 * np.pi * theta / b[1])

    return result

# %% Function for generating data

def gen_rastrigin_data(phi, theta, a, b):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    freqs = np.empty_like(x)

    n_rows, n_cols = freqs.shape

    for i in range(n_rows):
        for j in range(n_cols):
            freqs[i, j] = rastrigin_function(phi[i], theta[j], a, b)

    return x, y, z, freqs

# %% Generate data

n_incl = 100 # 25

# Inclination theta and azimuth phi
phi = np.linspace(-np.pi, np.pi, num=2 * n_incl + 1, endpoint=True)
theta = np.tile(np.linspace(0, np.pi, num=n_incl, endpoint=True), 2)

a = 10.
b = [0.01, 0.02] # [10 * np.pi, 10 * np.pi]

x, y, z, freqs = gen_rastrigin_data(phi, theta, a, b)

freqs[-1] = freqs[0]

n_samples = (freqs.shape[0] - 1) * freqs.shape[1]

# %% Generate training and test IDs

ids = np.arange(n_samples)

n_train = int(0.5 * n_samples)

train_ids = np.random.choice(ids, size=n_train, replace=False)

train_ids.sort()

test_ids = np.array(list(set(ids).difference(set(train_ids))))

test_ids.sort()

# %% Save data

data_path.mkdir(parents=True, exist_ok=True)

np.savetxt(data_path.joinpath('phi.csv'), phi)
np.savetxt(data_path.joinpath('theta.csv'), theta)

np.savetxt(data_path.joinpath('x.csv'), x, delimiter=',')
np.savetxt(data_path.joinpath('y.csv'), y, delimiter=',')
np.savetxt(data_path.joinpath('z.csv'), z, delimiter=',')

np.savetxt(data_path.joinpath('freqs.csv'), freqs, delimiter=',')

np.savetxt(data_path.joinpath('train_ids.csv'), train_ids, fmt='%i')
np.savetxt(data_path.joinpath('test_ids.csv'), test_ids, fmt='%i')
