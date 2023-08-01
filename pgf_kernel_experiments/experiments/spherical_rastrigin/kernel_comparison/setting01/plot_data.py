# %% Import packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.spherical_rastrigin.kernel_comparison.setting01.set_env import data_path, output_path
from pgf_kernel_experiments.plots import set_axes_equal

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)

grid = data[:, 2:5]
x = data[:, 2]
y = data[:, 3]
z = data[:, 4]
v = data[:, 5]

dims = np.loadtxt(data_path.joinpath('dims.csv'), dtype='int')

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %% Get training data

train_pos = grid[train_ids, :]

train_output = v[train_ids]

# %% Get test data

test_pos = grid[test_ids, :]

test_output = v[test_ids]

# %% Reshape data for plotting

x_plot = x.reshape(dims[0], dims[1], order='C')
x_plot = np.vstack([x_plot, x_plot[0, :]])

y_plot = y.reshape(dims[0], dims[1], order='C')
y_plot = np.vstack([y_plot, y_plot[0, :]])

z_plot = z.reshape(dims[0], dims[1], order='C')
z_plot = np.vstack([z_plot, z_plot[0, :]])

v_plot = v.reshape(dims[0], dims[1], order='C')
v_plot = np.vstack([v_plot, v_plot[0, :]])

# %% Generate  plot points for training data

train_v_plot = v_plot.copy()
train_v_plot = train_v_plot.flatten()
train_v_plot[test_ids] = np.nan
train_v_plot = train_v_plot.reshape(*(v_plot.shape))

# %% Generate plot points for test data

test_v_plot = v_plot.copy()
test_v_plot = test_v_plot.flatten()
test_v_plot[train_ids] = np.nan
test_v_plot = test_v_plot.reshape(*(v_plot.shape))

# %% Plot data, including separate training and test data (adding color map)

title_fontsize = 15
colorbar_fontsize = 11

fig = plt.figure(figsize=[14, 6])

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_adjust.html
# https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
# https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/

fig.subplots_adjust(
    left=-0.1,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.02, # -0.65,
    hspace=0.0 # 0.15
)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

norm = plt.Normalize()

# https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow-in-matplotlib

cmap = plt.cm.jet
cmap.set_bad('white')

# https://github.com/matplotlib/matplotlib/issues/14647

ax1.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(v_plot)), edgecolor='none')

ax1.set_title('All data', fontsize=title_fontsize)

ax1.set_box_aspect([1, 1, 1])

# ax1.set_proj_type('ortho') # default is perspective

set_axes_equal(ax1)

ax1.grid(False)
ax1.axis('off')

xyz_lim = 0.63

ax1.set_xlim(-xyz_lim, xyz_lim)
ax1.set_ylim(-xyz_lim, xyz_lim)
ax1.set_zlim(-xyz_lim, xyz_lim)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')

ax2.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(train_v_plot)), edgecolor='none')

ax2.set_title('Training data', fontsize=title_fontsize)

ax2.set_box_aspect([1, 1, 1])

# ax2.set_proj_type('ortho') # default is perspective

set_axes_equal(ax2)

ax2.grid(False)
ax2.axis('off')

ax2.set_xlim(-xyz_lim, xyz_lim)
ax2.set_ylim(-xyz_lim, xyz_lim)
ax2.set_zlim(-xyz_lim, xyz_lim)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax3.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(test_v_plot)), edgecolor='none')

ax3.set_title('Test data', fontsize=title_fontsize)

ax3.set_box_aspect([1, 1, 1])

# ax3.set_proj_type('ortho') # default is perspective

set_axes_equal(ax3)

ax3.grid(False)
ax3.axis('off')

ax3.set_xlim(-xyz_lim, xyz_lim)
ax3.set_ylim(-xyz_lim, xyz_lim)
ax3.set_zlim(-xyz_lim, xyz_lim)

# https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/

fig.subplots_adjust(bottom=0.0, right=0.80, top=1.0)

cax = fig.add_axes([0.80, 0.2, 0.01, 0.6])

# https://stackoverflow.com/questions/33443334/how-to-decrease-colorbar-width-in-matplotlib
# https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), cax=cax) # , aspect=30)

# https://stackoverflow.com/questions/69435068/change-colorbar-limit-for-changing-scale-with-matplotlib-3-3

# cb.mappable.set_clim(0., 60.)

# https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/

# cb_tick_points = np.arange(0, 60+10, 10)
# cb.set_ticks(cb_tick_points)
# cb.set_ticklabels(cb_tick_points)

# https://www.tutorialspoint.com/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbase

cb.ax.tick_params(labelsize=colorbar_fontsize)

# plt.show()

# %% Plot a histogram of spherical Rastrigin function values

plt.hist(v_plot.flatten(), bins=100, density=True)

# %% Save plot

fig.savefig(
    output_path.joinpath('data.png'),
    dpi=600,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close(fig)
