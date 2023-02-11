# %% Import packages

import numpy as np

from bloch_density import BlochDensity

# %% Set seed

np.random.seed(7)

# %% Load data

phi = np.loadtxt('phi.csv')
theta = np.loadtxt('theta.csv')

x = np.loadtxt('x.csv', delimiter=',')
y = np.loadtxt('y.csv', delimiter=',')
z = np.loadtxt('z.csv', delimiter=',')

freqs = np.loadtxt('freqs.csv', delimiter=',')
