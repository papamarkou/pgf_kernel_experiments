# %% Import packages

import numpy as np

from scipy.special import gamma

# %% Class for Cox process

# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
# https://math.stackexchange.com/questions/56582/what-is-the-analogue-of-spherical-coordinates-in-n-dimensions

class CoxProcess:
    def __init__(self, n, r, lambdas, scales):
        self.n = n
        self.r = r
        self.lambdas = lambdas
        self.scales = scales

    def num_clusters(self):
        return len(self.lambdas)

    def surface_area(self):
        return 2 * (np.pi^(self.n/2)) / gamma(self.n/2)

    def poisson_rates(self):
        return self.surface_area() * self.lambdas

    def simulate_cluster_centers(self):
        pass

    def spherical_to_polar_coords(self):
        pass

# %%

def simulate_cox_process(n, r, intensities, scales):
    num_clusters = len(intensities)

    center_spherical_coords = None

    center_cartesian_coords = None

    # np.random.default_rng().normal(loc=0.0, scale=0.2, size=v_signal.shape)

    data_polar_coords = None

    data_cartesian_coords = None

    labels = None

    return data_polar_coords, data_cartesian_coords, labels, center_spherical_coords, center_cartesian_coords

# %% Function for computing discoball function given input in Cartesian coordinates

def discoball_function(phi, theta, k, l, m, terms, a=1., b=0.):
    num_terms = len(k)

    exponent = 0.
    for i in range(num_terms):
        if terms[i] == 0:
            angle = phi
        elif terms[i] == 1:
            angle = theta
        elif terms[i] == 2:
            angle = phi + theta
        exponent += k[i] * np.cos(l[i] * (angle + m[i]))

    return a * np.exp(exponent) + b

# %% Function for generating data from the discoball function given polar coordinates

def gen_discoball_data(phi, theta, k, l, m, terms, a=1., b=0.):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    phi_grid = np.outer(phi, np.ones(np.size(theta)))
    theta_grid = np.outer(np.ones(np.size(phi)), theta)

    v = np.empty_like(x)

    n_rows, n_cols = v.shape

    for i in range(n_rows):
        for j in range(n_cols):
            v[i, j] = discoball_function(phi_grid[i, j], theta_grid[i, j], k, l, m, terms, a=a, b=b)

    return x, y, z, v
