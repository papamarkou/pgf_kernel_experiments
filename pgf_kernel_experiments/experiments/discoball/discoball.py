# %% Import packages

import numpy as np

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
