# %% Import packages

import numpy as np

# %% Function for computing spherical Rastrigin function given input in polar coordinates

def spherical_rastrigin_function(phi, theta, a, b):
    result = 2 * a
    # result = result + ((phi / b[0]) ** 2) - a * np.cos(2 * np.pi * phi / b[0])
    # result = result + ((theta / b[1]) ** 2) - a * np.cos(2 * np.pi * theta / b[1])
    result = result + (phi ** 2) - a * np.cos(2 * np.pi * phi / b[0])
    result = result + (theta ** 2) - a * np.cos(2 * np.pi * theta / b[1])

    return result

# %% Function for generating data from the spherical Rastrigin function

def gen_spherical_rastrigin_data(phi, theta, a, b):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    v = np.empty_like(x)

    n_rows, n_cols = v.shape

    for i in range(n_rows):
        for j in range(n_cols):
            v[i, j] = spherical_rastrigin_function(phi[i], theta[j], a, b)

    return x, y, z, v
