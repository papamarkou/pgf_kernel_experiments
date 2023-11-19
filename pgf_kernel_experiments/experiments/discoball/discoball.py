# %% Import packages

import numpy as np

# %% Function for computing trigonometric function given input in Cartesian coordinates

# https://www.chebfun.org/docs/guide/guide17.html

# def trigonometric_function(x, y, z, a=1.):
#     return a * np.cos(np.cosh(5 * x * z) - 10 * y)

def trigonometric_function(phi, theta, k, l, a=1., b=0.):
    # return a * np.exp(np.cos(10 * (x + y + z)))
    # return a * np.exp(1.5 * np.cos(15 * phi) + 1.5 * np.cos(15 * theta)) + b
    return a * np.exp(k[0] * np.cos(l[0] * phi) + k[1] * np.cos(l[1] * theta) + k[2] * np.sin(l[2] * (phi + theta))) + b
    # return a * np.exp(x + y + z)

# %% Function for generating data from the trigonometric function given polar coordinates

# def gen_trigonometric_data(phi, theta, a=1.):
#     x = np.outer(np.cos(phi), np.sin(theta))
#     y = np.outer(np.sin(phi), np.sin(theta))
#     z = np.outer(np.ones(np.size(phi)), np.cos(theta))

#     v = np.empty_like(x)

#     n_rows, n_cols = v.shape

#     for i in range(n_rows):
#         for j in range(n_cols):
#             v[i, j] = trigonometric_function(x[i, j], y[i, j], z[i, j], a=a)

#     return x, y, z, v

def gen_trigonometric_data(phi, theta, k, l, a=1., b=0.):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    phi_grid = np.outer(phi, np.ones(np.size(theta)))
    theta_grid = np.outer(np.ones(np.size(phi)), theta)

    v = np.empty_like(x)

    n_rows, n_cols = v.shape

    for i in range(n_rows):
        for j in range(n_cols):
            v[i, j] = trigonometric_function(phi_grid[i, j], theta_grid[i, j], k, l, a=a, b=b)

    return x, y, z, v
