# %% Import packages

import numpy as np

from bloch_density import BlochDensity
from set_paths import data_path

# %% Set seed

np.random.seed(7)

# %% Load data

phi = np.loadtxt(data_path.joinpath('phi.csv'))
theta = np.loadtxt(data_path.joinpath('theta.csv'))

x = np.loadtxt(data_path.joinpath('x.csv'), delimiter=',')
y = np.loadtxt(data_path.joinpath('y.csv'), delimiter=',')
z = np.loadtxt(data_path.joinpath('z.csv'), delimiter=',')

freqs = np.loadtxt(data_path.joinpath('freqs.csv'), delimiter=',')

n_samples = freqs.shape[0] * freqs.shape[1]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %%

phi_front = phi[:int(freqs.shape[0] / 2)]
theta_front = theta
x_front = x[:int(freqs.shape[0] / 2), :]
y_front = y[:int(freqs.shape[0] / 2), :]
z_front = z[:int(freqs.shape[0] / 2), :]
freqs_front = freqs[:int(freqs.shape[0] / 2), :]

phi_back = phi[(int(freqs.shape[0] / 2) - 1):]
theta_back = theta
x_back = x[(int(freqs.shape[0] / 2) - 1):, :]
y_back = y[(int(freqs.shape[0] / 2) - 1):, :]
z_back = z[(int(freqs.shape[0] / 2) - 1):, :]
freqs_back = freqs[(int(freqs.shape[0] / 2) - 1):, :]

b = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, freqs_front,
    phi_back, theta_back, x_back, y_back, z_back, freqs_back,
    alpha = 0.33
)

# %%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=[16, 6]) # constrained_layout=True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(10), range(10), "o-")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# b1 = Bloch(fig=fig, axes=ax2)
b.fig = fig
b.axes = ax2
b.render()
ax2.set_box_aspect([1, 1, 1]) # required for mpl > 3.1

plt.show()

# %%
