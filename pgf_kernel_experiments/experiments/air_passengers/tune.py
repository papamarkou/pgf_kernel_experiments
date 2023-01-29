# %% Import packages

import numpy as np

# %% Load data

dataset = np.loadtxt('passenger_numbers.csv')

# %% Plot data

import matplotlib.pyplot as plt

plt.plot(dataset)

# %% Place data on the unit circle

n_samples = len(dataset)

theta = np.linspace(0, 2*np.pi, num=n_samples, endpoint=False)

x = np.cos(theta)

y = np.sin(theta)

grid = np.stack((x, y))

z = dataset

