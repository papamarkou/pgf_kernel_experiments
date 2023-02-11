# %% Import packages

import numpy as np
import scipy

from pathlib import Path

# %% Set seed

np.random.seed(7)

# %% Function for generating uniform polar density

def gen_unif_polar_density(phi, theta):
    return scipy.interpolate.RegularGridInterpolator((phi, theta), np.random.rand(len(phi), len(theta)))

# %% Function for generating data

def gen_bloch_data(phi, theta):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    freqs = np.empty_like(x)

    n_rows, n_cols = freqs.shape

    for i in range(n_rows):
        for j in range(n_cols):
            freqs[i, j] = unif_polar_density(np.array([phi[i], theta[j]]))

    return x, y, z, freqs

# %% Generate uniform polar density

n_design = 20

# Inclination theta and azimuth phi
phi_design, theta_design = np.linspace(-np.pi, np.pi, n_design), np.linspace(0, np.pi, n_design)

unif_polar_density = gen_unif_polar_density(phi_design, theta_design)

# %% Generate data

n_incl = 25

phi = np.linspace(-np.pi, np.pi, 2 * n_incl)
theta = np.tile(np.linspace(0, np.pi, n_incl), 2)

x, y, z, freqs = gen_bloch_data(phi, theta)

# %% Save data

data_path = Path('data')

data_path.mkdir(parents=True, exist_ok=True)

np.savetxt(data_path.joinpath('phi.csv'), phi)
np.savetxt(data_path.joinpath('theta.csv'), theta)

np.savetxt(data_path.joinpath('x.csv'), x, delimiter=',')
np.savetxt(data_path.joinpath('y.csv'), y, delimiter=',')
np.savetxt(data_path.joinpath('z.csv'), z, delimiter=',')

np.savetxt(data_path.joinpath('freqs.csv'), freqs, delimiter=',')
