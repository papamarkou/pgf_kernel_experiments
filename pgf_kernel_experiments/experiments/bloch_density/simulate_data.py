# %% Import packages

import numpy as np
import scipy

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

    return x, y, z, unif_polar_density(np.column_stack((phi, theta)))

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

np.savetxt('phi.csv', phi)
np.savetxt('theta.csv', theta)

np.savetxt('x.csv', x, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')
np.savetxt('z.csv', z, delimiter=',')

np.savetxt('freqs.csv', freqs, delimiter=',')
