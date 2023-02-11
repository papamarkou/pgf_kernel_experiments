# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from bloch_density import BlochDensity
from set_paths import data_path

# %% Load data

phi = np.loadtxt(data_path.joinpath('phi.csv'))
theta = np.loadtxt(data_path.joinpath('theta.csv'))

x = np.loadtxt(data_path.joinpath('x.csv'), delimiter=',')
y = np.loadtxt(data_path.joinpath('y.csv'), delimiter=',')
z = np.loadtxt(data_path.joinpath('z.csv'), delimiter=',')

freqs = np.loadtxt(data_path.joinpath('freqs.csv'), delimiter=',')

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %% Set up training and test data

# %% Generate BlochDensity for all data

n_train_freqs = int((freqs.shape[0] - 1) / 2)

phi_front = phi[:n_train_freqs]
theta_front = theta
x_front = x[:n_train_freqs, :]
y_front = y[:n_train_freqs, :]
z_front = z[:n_train_freqs, :]
freqs_front = freqs[:n_train_freqs, :]

phi_back = phi[(n_train_freqs - 1):]
theta_back = theta
x_back = x[(n_train_freqs - 1):, :]
y_back = y[(n_train_freqs - 1):, :]
z_back = z[(n_train_freqs - 1):, :]
freqs_back = freqs[(n_train_freqs - 1):, :]

bloch_all_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, freqs_front,
    phi_back, theta_back, x_back, y_back, z_back, freqs_back,
    alpha = 0.33
)

# %% Generate BlochDensity for training data

bloch_training_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, freqs_front,
    phi_back, theta_back, x_back, y_back, z_back, freqs_back,
    alpha = 0.33
)

# %% Generate BlochDensity for test data

bloch_test_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, freqs_front,
    phi_back, theta_back, x_back, y_back, z_back, freqs_back,
    alpha = 0.33
)

# %% Plot data

# https://qutip.org/docs/4.0.2/guide/guide-bloch.html

fontsize = 18

fig = plt.figure(figsize=[16, 6], constrained_layout=True)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

bloch_all_data.fig = fig
bloch_all_data.axes = ax1

bloch_all_data.xlpos = [1.55, -1.1]
bloch_all_data.zlpos = [1.22, -1.35]

bloch_all_data.render()

ax1.set_box_aspect([1, 1, 1]) 

ax1.set_title('All data', fontsize=fontsize)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')

bloch_training_data.fig = fig
bloch_training_data.axes = ax2

bloch_training_data.xlpos = [1.55, -1.1]
bloch_training_data.zlpos = [1.22, -1.35]

bloch_training_data.render()

ax2.set_box_aspect([1, 1, 1])

ax2.set_title('Training data', fontsize=fontsize)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')

bloch_test_data.fig = fig
bloch_test_data.axes = ax3

bloch_test_data.xlpos = [1.55, -1.1]
bloch_test_data.zlpos = [1.22, -1.35]

bloch_test_data.render()

ax3.set_box_aspect([1, 1, 1])

ax3.set_title('Test data', fontsize=fontsize)

# plt.show()

# %%
