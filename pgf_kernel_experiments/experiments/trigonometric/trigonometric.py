# %% Import packages

import numpy as np

# %% Function for computing trigonometric function given input in Cartesian coordinates

# https://www.chebfun.org/docs/guide/guide17.html

def trigonometric_function(x, y, z):
    return np.cos(np.cosh(5 * x * z) - 10 * y)

# %% Function for generating data from the trigonometric function given polar coordinates

def gen_trigonometric_data(phi, theta):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    v = np.empty_like(x)

    n_rows, n_cols = v.shape

    for i in range(n_rows):
        for j in range(n_cols):
            v[i, j] = trigonometric_function(x[i, j], y[i, j], z[i, j])

    return x, y, z, v
