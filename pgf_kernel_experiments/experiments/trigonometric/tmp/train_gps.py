# %% Import packages

import gpytorch
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colorbar import ColorbarBase, make_axes_gridspec

from pgf_kernel_experiments.runners import ExactMultiGPRunner
from pgfml.kernels import GFKernel

# from pgf_kernel_experiments.experiments.bloch_density.bloch_density import BlochDensity
from pgf_kernel_experiments.experiments.trigonometric.tmp.set_paths import data_path, output_path

# %% Create paths if they don't exist

data_path.mkdir(parents=True, exist_ok=True)
output_path.mkdir(parents=True, exist_ok=True)

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

# %% Functions for setting up equal aspect ratio

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

# %% Plot data

fontsize = 18

fig = plt.figure(figsize=[8, 6], constrained_layout=True)

ax1 = fig.add_subplot(1, 1, 1, projection='3d') #, aspect='equal')

norm = plt.Normalize()

ax1.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=plt.cm.jet(norm(freqs)), edgecolor='none')

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

ax1.set_box_aspect([1, 1, 1])

# https://www.tutorialspoint.com/differentiate-the-orthographic-and-perspective-projection-in-matplotlib

# ax1.set_proj_type('ortho') # default is perspective

set_axes_equal(ax1)

# https://www.tutorialspoint.com/how-to-hide-axes-and-gridlines-in-matplotlib

ax1.grid(False)
ax1.axis('off')

# https://stackoverflow.com/questions/41225293/remove-white-spaces-in-axes3d-matplotlib

xyz_lim = 0.63

ax1.set_xlim(-xyz_lim, xyz_lim)
ax1.set_ylim(-xyz_lim, xyz_lim)
ax1.set_zlim(-xyz_lim, xyz_lim)

# https://stackoverflow.com/questions/33569225/attaching-intensity-to-3d-plot

cax, kw = make_axes_gridspec(ax1, shrink=0.6, aspect=20)
cb = ColorbarBase(cax, cmap=plt.cm.jet, norm=norm)
# cb.set_label('Value', fontsize='x-large')

# https://stackoverflow.com/questions/69435068/change-colorbar-limit-for-changing-scale-with-matplotlib-3-3

cb.mappable.set_clim(0., 60.)

# https://www.tutorialspoint.com/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbase

cb.ax.tick_params(labelsize=fontsize)

# %% Set up training and test data

train_pos = pos[train_ids, :]

train_output = freqs_flat[train_ids]

test_pos = pos[test_ids, :]

test_output = freqs_flat[test_ids]

# %% Generate  plot points for training data

train_freqs_plot = freqs.copy()
train_freqs_plot = train_freqs_plot.flatten()
train_freqs_plot[test_ids] = np.nan
train_freqs_plot = train_freqs_plot.reshape(*(freqs.shape))

# %% Generate plot points for test data

test_freqs_plot = freqs.copy()
test_freqs_plot = test_freqs_plot.flatten()
test_freqs_plot[train_ids] = np.nan
test_freqs_plot = test_freqs_plot.reshape(*(freqs.shape))

# %% Plot data, including separate training and test data (adding color map)

fontsize = 18

fig = plt.figure(figsize=[14, 6])

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_adjust.html
# https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
# https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.65,
    hspace=0.15
)

ax1 = fig.add_subplot(2, 3, 1, projection='3d')

norm = plt.Normalize()

# https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow-in-matplotlib

cmap = plt.cm.jet
cmap.set_bad('white')

# https://github.com/matplotlib/matplotlib/issues/14647

ax1.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(freqs)), edgecolor='none')

ax1.set_title('All data', fontsize=fontsize)

ax1.set_box_aspect([1, 1, 1])

# ax1.set_proj_type('ortho') # default is perspective

set_axes_equal(ax1)

ax1.grid(False)
ax1.axis('off')

xyz_lim = 0.63

ax1.set_xlim(-xyz_lim, xyz_lim)
ax1.set_ylim(-xyz_lim, xyz_lim)
ax1.set_zlim(-xyz_lim, xyz_lim)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')

ax2.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(train_freqs_plot)), edgecolor='none')

ax2.set_title('Training data', fontsize=fontsize)

ax2.set_box_aspect([1, 1, 1])

# ax2.set_proj_type('ortho') # default is perspective

set_axes_equal(ax2)

ax2.grid(False)
ax2.axis('off')

ax2.set_xlim(-xyz_lim, xyz_lim)
ax2.set_ylim(-xyz_lim, xyz_lim)
ax2.set_zlim(-xyz_lim, xyz_lim)

ax3 = fig.add_subplot(2, 3, 3, projection='3d')

ax3.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(test_freqs_plot)), edgecolor='none')

ax3.set_title('Test data', fontsize=fontsize)

ax3.set_box_aspect([1, 1, 1])

# ax3.set_proj_type('ortho') # default is perspective

set_axes_equal(ax3)

ax3.grid(False)
ax3.axis('off')

ax3.set_xlim(-xyz_lim, xyz_lim)
ax3.set_ylim(-xyz_lim, xyz_lim)
ax3.set_zlim(-xyz_lim, xyz_lim)

ax4 = fig.add_subplot(2, 3, 4, projection='3d')

ax4.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(test_freqs_plot)), edgecolor='none')

ax4.set_title('Test data', fontsize=fontsize)

ax4.set_box_aspect([1, 1, 1])

# ax4.set_proj_type('ortho') # default is perspective

set_axes_equal(ax4)

ax4.grid(False)
ax4.axis('off')

ax4.set_xlim(-xyz_lim, xyz_lim)
ax4.set_ylim(-xyz_lim, xyz_lim)
ax4.set_zlim(-xyz_lim, xyz_lim)

ax5 = fig.add_subplot(2, 3, 5, projection='3d')

ax5.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(test_freqs_plot)), edgecolor='none')

ax5.set_title('Test data', fontsize=fontsize)

ax5.set_box_aspect([1, 1, 1])

# ax5.set_proj_type('ortho') # default is perspective

set_axes_equal(ax5)

ax5.grid(False)
ax5.axis('off')

ax5.set_xlim(-xyz_lim, xyz_lim)
ax5.set_ylim(-xyz_lim, xyz_lim)
ax5.set_zlim(-xyz_lim, xyz_lim)

ax6 = fig.add_subplot(2, 3, 6, projection='3d')

ss = ax6.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap(norm(test_freqs_plot)), edgecolor='none')

ax6.set_title('Test data', fontsize=fontsize)

ax6.set_box_aspect([1, 1, 1])

# ax6.set_proj_type('ortho') # default is perspective

set_axes_equal(ax6)

ax6.grid(False)
ax6.axis('off')

ax6.set_xlim(-xyz_lim, xyz_lim)
ax6.set_ylim(-xyz_lim, xyz_lim)
ax6.set_zlim(-xyz_lim, xyz_lim)

# https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/

fig.subplots_adjust(bottom=0.0, right=0.97, top=1.0)

cax = fig.add_axes([0.80, 0.1, 0.01, 0.8])

# https://stackoverflow.com/questions/33443334/how-to-decrease-colorbar-width-in-matplotlib
# https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), cax=cax) # , aspect=30)

# https://stackoverflow.com/questions/69435068/change-colorbar-limit-for-changing-scale-with-matplotlib-3-3

cb.mappable.set_clim(0., 60.)

# https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/

cb_tick_points = np.arange(0, 60+10, 10)
cb.set_ticks(cb_tick_points)
cb.set_ticklabels(cb_tick_points)

# https://www.tutorialspoint.com/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbase

cb.ax.tick_params(labelsize=fontsize)

# plt.show()

# %%

fig.savefig(
    output_path.joinpath('trigonometric_predictions.png'),
    dpi=600,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()

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
