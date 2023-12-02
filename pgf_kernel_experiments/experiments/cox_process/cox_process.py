# %% Import packages

import numpy as np

from scipy.special import gamma

# %% Class for coordinate system

# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
# https://math.stackexchange.com/questions/56582/what-is-the-analogue-of-spherical-coordinates-in-n-dimensions

class CoordSys:
    def __init__(self, n=None):
        self.n = n

    def simulate_spherical_coords(self, num_points):
        spherical_coords = np.empty([num_points, self.n])

        spherical_coords[:, :(self.n-1)] = np.random.default_rng().uniform(low=0, high=np.pi, size=[num_points, self.n-1])
        spherical_coords[:, self.n-1] = np.random.default_rng().uniform(low=0, high=2*np.pi, size=num_points)

        return spherical_coords

    def spherical_to_cartesian_coords(self, spherical_coords):
        num_points, n = spherical_coords.shape
        assert self.n == n

        cartesian_coords = np.empty([num_points, self.n+1])

        cartesian_coords[:, 0] = np.cos(spherical_coords[:, 0])

        sin_products = 1

        for i in range(1, self.n):
            sin_products = sin_products * np.sin(spherical_coords[:, i-1])
            cartesian_coords[:, i] = sin_products * np.cos(spherical_coords[:, i])

        cartesian_coords[:, self.n] = sin_products * np.sin(spherical_coords[:, self.n-1])

        return cartesian_coords

# %%

coord_sys = CoordSys(4)

spherical_coords = coord_sys.simulate_spherical_coords(2)

cartesian_coords = coord_sys.spherical_to_cartesian_coords(spherical_coords)

print(np.linalg.norm(cartesian_coords, axis=1))

# %% Calss for Cox process

class CoxProcess:
    def __init__(self, n, lambdas, scales):
        self.n = n
        self.lambdas = lambdas
        self.scales = scales

    def get_num_clusters(self):
        return len(self.lambdas)

    def get_surface_area(self):
        return 2 * (np.pi^(self.n/2)) / gamma(self.n/2)

    def get_poisson_rates(self):
        return self.surface_area() * self.lambdas

    def simulate_cluster_centers(self):
        pass

    def simulate_num_points(self):
        pass

    def simulate_data(self):
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
