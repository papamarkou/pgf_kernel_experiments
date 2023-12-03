# %% Import packages

import numpy as np

from scipy.special import gamma
from scipy.stats import vonmises_fisher

# %% Class for coordinate system

# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
# https://math.stackexchange.com/questions/56582/what-is-the-analogue-of-spherical-coordinates-in-n-dimensions

class CoordSys:
    def __init__(self, n=None):
        self.n = n

    def get_num_spherical_coords(self):
        return self.n

    def get_num_cartesian_coords(self):
        return self.get_num_spherical_coords() + 1

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

# %% Calss for Cox process

class CoxProcess:
    def __init__(self, n, lambdas, kappas):
        self.coord_sys = CoordSys(n)
        self.lambdas = np.array(lambdas)
        self.kappas = np.array(kappas)

    def get_num_clusters(self):
        return len(self.lambdas)

    def get_surface_area(self):
        return 2 * (np.pi ** (self.coord_sys.n / 2)) / gamma(self.coord_sys.n / 2)

    def get_poisson_rates(self):
        return self.get_surface_area() * self.lambdas

    def simulate_cluster_centers(self):
        return self.coord_sys.simulate_spherical_coords(self.get_num_clusters())

    def simulate_num_points(self):
        return np.random.default_rng().poisson(lam=self.get_surface_area() * self.lambdas)

    def simulate_data(self, cluster_centers=None, num_points=None):
        if cluster_centers is None:
            center_spherical_coords = self.simulate_cluster_centers()
            center_cartesian_coords = self.coord_sys.spherical_to_cartesian_coords(center_spherical_coords)
        else:
            center_cartesian_coords = cluster_centers

        if num_points is None:
            num_points = self.simulate_num_points()

        data_cartesian_coords = []

        for i in range(self.get_num_clusters()):
            data_cartesian_coords.append(vonmises_fisher(
                center_cartesian_coords[i, :], self.kappas[i]).rvs(num_points[i], random_state=np.random.default_rng()
            ))

        data_cartesian_coords = np.vstack(data_cartesian_coords)

        labels = np.repeat(list(range(len(num_points))), num_points)

        return data_cartesian_coords, labels, center_cartesian_coords, num_points
