# %% Import packages

import numpy as np

# %% Function for amplitude-phase form of Fourier series

def fourier_series(theta, a, p, phi):
    result = np.full_like(theta, 0.5 * a[0])

    for n in range(1, len(a)):
        result = result + a[n] * np.cos(2 * np.pi * n * theta / p - phi[n-1])

    return result
