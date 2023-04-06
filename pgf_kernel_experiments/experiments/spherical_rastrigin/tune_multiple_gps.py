# %% Import packages

import gpytorch
import torch

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.runners import ExactMultiGPRunner
from pgfml.kernels import GFKernel

from pgf_kernel_experiments.experiments.bloch_density.bloch_density import BlochDensity
from pgf_kernel_experiments.experiments.bloch_density.set_paths import data_path

# %% Load data

phi = np.loadtxt(data_path.joinpath('phi.csv'))
theta = np.loadtxt(data_path.joinpath('theta.csv'))

x = np.loadtxt(data_path.joinpath('x.csv'), delimiter=',')
y = np.loadtxt(data_path.joinpath('y.csv'), delimiter=',')
z = np.loadtxt(data_path.joinpath('z.csv'), delimiter=',')

freqs = np.loadtxt(data_path.joinpath('freqs.csv'), delimiter=',')

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

x_flat = x[:-1, :].flatten()
y_flat = y[:-1, :].flatten()
z_flat = z[:-1, :].flatten()

freqs_flat = freqs[:-1, :].flatten()

pos = np.column_stack((x_flat, y_flat, z_flat))

# %% Plot data

# https://qutip.org/docs/4.0.2/guide/guide-bloch.html

fontsize = 18

fig = plt.figure(figsize=[16, 6], constrained_layout=True)

ax1 = fig.add_subplot(1, 1, 1, projection='3d') #, aspect='equal')
# ax1.set_aspect('equal')
# ax1.set_axis_off()

norm = plt.Normalize()

ax1.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=plt.cm.jet(norm(freqs)))
# ax1.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=plt.cm.jet(freqs))

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

ax1.set_box_aspect([1, 1, 1])

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

# ax1.plot_surface(x, y, z, facecolors=plt.cm.jet(norm(freqs)))

# https://www.tutorialspoint.com/differentiate-the-orthographic-and-perspective-projection-in-matplotlib

# ax1.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)

set_axes_equal(ax1)

from matplotlib.colorbar import ColorbarBase, make_axes_gridspec

# https://stackoverflow.com/questions/33569225/attaching-intensity-to-3d-plot

cax, kw = make_axes_gridspec(ax1, shrink=0.6, aspect=15)
cb = ColorbarBase(cax, cmap=plt.cm.jet, norm=norm)
# cb.set_label('Value', fontsize='x-large')

# https://www.tutorialspoint.com/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbase

cb.ax.tick_params(labelsize=12)

# %%

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
# ax.hold(True)
ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=plt.cm.jet(norm(freqs)))

# %% Set up training and test data

train_pos = pos[train_ids, :]

train_output = freqs_flat[train_ids]

test_pos = pos[test_ids, :]

test_output = freqs_flat[test_ids]

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

train_freqs_plot = freqs.copy()
train_freqs_plot = train_freqs_plot.flatten()
train_freqs_plot[test_ids] = np.nan
train_freqs_plot = train_freqs_plot.reshape(*(freqs.shape))

train_freqs_plot_front = train_freqs_plot[:n_train_freqs, :]

train_freqs_plot_back = train_freqs_plot[(n_train_freqs - 1):, :]

bloch_train_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, train_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, train_freqs_plot_back,
    alpha = 0.33
)

# %% Generate BlochDensity for test data

test_freqs_plot = freqs.copy()
test_freqs_plot = test_freqs_plot.flatten()
test_freqs_plot[train_ids] = np.nan
test_freqs_plot = test_freqs_plot.reshape(*(freqs.shape))

test_freqs_plot_front = test_freqs_plot[:n_train_freqs, :]

test_freqs_plot_back = test_freqs_plot[(n_train_freqs - 1):, :]

bloch_test_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, test_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, test_freqs_plot_back,
    alpha = 0.33
)

# %% Plot data

# https://qutip.org/docs/4.0.2/guide/guide-bloch.html

fontsize = 18

fig = plt.figure(figsize=[16, 6], constrained_layout=True)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

norm = plt.Normalize()

ax1.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=plt.cm.jet(norm(freqs)))
# ax1.plot_surface(x, y, z, facecolors=plt.cm.jet(norm(freqs)))

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

bloch_train_data.fig = fig
bloch_train_data.axes = ax2

bloch_train_data.xlpos = [1.55, -1.1]
bloch_train_data.zlpos = [1.22, -1.35]

bloch_train_data.render()

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

# %% Convert training and test data to PyTorch format

train_x = torch.as_tensor(train_pos, dtype=torch.float64)
train_y = torch.as_tensor(train_output, dtype=torch.float64)

test_x = torch.as_tensor(test_pos, dtype=torch.float64)
test_y = torch.as_tensor(test_output, dtype=torch.float64)

# %% Set up ExactMultiGPRunner

kernels = [
    GFKernel(width=[20, 20, 20]),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
    gpytorch.kernels.PeriodicKernel(),
    # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=3)
]

runner = ExactMultiGPRunner.generator(train_x, train_y, kernels)

# %% Set the models in double mode

for i in range(len(kernels)):
    runner.single_runners[i].model.double()
    runner.single_runners[i].model.likelihood.double()

# %% Configurate training setup for GP models

optimizers = []

for i in range(runner.num_gps()):
    optimizers.append(torch.optim.Adam(runner.single_runners[i].model.parameters(), lr=0.1))

n_iters = 10

# %% Train GP models to find optimal hyperparameters

losses = runner.train(train_x, train_y, optimizers, n_iters)

# %% Make predictions

predictions = runner.test(test_x)

# %% Compute error metrics

scores = runner.assess(
    predictions,
    test_y,
    metrics=[
        gpytorch.metrics.mean_absolute_error,
        gpytorch.metrics.mean_squared_error,
        lambda predictions, y : -gpytorch.metrics.negative_log_predictive_density(predictions, y)
    ]
)

# %% Generate BlochDensity for PGF-GP predictions

pgf_freqs_plot = np.empty_like(freqs)
pgf_freqs_plot = pgf_freqs_plot.flatten()
pgf_freqs_plot[train_ids] = np.nan
pgf_freqs_plot[test_ids] = predictions[0].mean
pgf_freqs_plot = pgf_freqs_plot.reshape(*(freqs.shape))

pgf_freqs_plot_front = pgf_freqs_plot[:n_train_freqs, :]

pgf_freqs_plot_back = pgf_freqs_plot[(n_train_freqs - 1):, :]

bloch_pgf_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, pgf_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, pgf_freqs_plot_back,
    alpha = 0.33
)

# %% Generate BlochDensity for RBF-GP predictions

rbf_freqs_plot = np.empty_like(freqs)
rbf_freqs_plot = rbf_freqs_plot.flatten()
rbf_freqs_plot[train_ids] = np.nan
rbf_freqs_plot[test_ids] = predictions[0].mean
rbf_freqs_plot = rbf_freqs_plot.reshape(*(freqs.shape))

rbf_freqs_plot_front = rbf_freqs_plot[:n_train_freqs, :]

rbf_freqs_plot_back = rbf_freqs_plot[(n_train_freqs - 1):, :]

bloch_rbf_data = BlochDensity(
    phi_front, theta_front, x_front, y_front, z_front, rbf_freqs_plot_front,
    phi_back, theta_back, x_back, y_back, z_back, rbf_freqs_plot_back,
    alpha = 0.33
)

# %% Plot predictions

fontsize = 18

fig = plt.figure(figsize=[16, 8], constrained_layout=True)

ax1 = fig.add_subplot(2, 4, 1, projection='3d')

bloch_all_data.fig = fig
bloch_all_data.axes = ax1

bloch_all_data.xlpos = [1.55, -1.1]
bloch_all_data.zlpos = [1.22, -1.35]

bloch_all_data.render()

ax1.set_box_aspect([1, 1, 1]) 

ax1.set_title('All data', fontsize=fontsize)

ax2 = fig.add_subplot(2, 4, 2, projection='3d')

bloch_train_data.fig = fig
bloch_train_data.axes = ax2

bloch_train_data.xlpos = [1.55, -1.1]
bloch_train_data.zlpos = [1.22, -1.35]

bloch_train_data.render()

ax2.set_box_aspect([1, 1, 1])

ax2.set_title('Training data', fontsize=fontsize)

ax3 = fig.add_subplot(2, 4, 3, projection='3d')

bloch_test_data.fig = fig
bloch_test_data.axes = ax3

bloch_test_data.xlpos = [1.55, -1.1]
bloch_test_data.zlpos = [1.22, -1.35]

bloch_test_data.render()

ax3.set_box_aspect([1, 1, 1])

ax3.set_title('Test data', fontsize=fontsize)

ax4 = fig.add_subplot(2, 4, 4, projection='3d')

bloch_pgf_data.fig = fig
bloch_pgf_data.axes = ax4

bloch_pgf_data.xlpos = [1.55, -1.1]
bloch_pgf_data.zlpos = [1.22, -1.35]

bloch_pgf_data.render()

ax4.set_box_aspect([1, 1, 1])

ax4.set_title('PGF-GP predictions', fontsize=fontsize)

ax5 = fig.add_subplot(2, 4, 5, projection='3d')

bloch_rbf_data.fig = fig
bloch_rbf_data.axes = ax5

bloch_rbf_data.xlpos = [1.55, -1.1]
bloch_rbf_data.zlpos = [1.22, -1.35]

bloch_rbf_data.render()

ax5.set_box_aspect([1, 1, 1])

ax5.set_title('RBF-GP predictions', fontsize=fontsize)

# plt.show()

# %%
