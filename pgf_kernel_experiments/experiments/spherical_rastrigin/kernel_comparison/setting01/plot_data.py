# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.spherical_rastrigin.kernel_comparison.setting01.set_env import data_path, output_path

# %% Create paths if they don't exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)

grid = data[:, 1:3]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')
test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %% Get training data

train_pos = grid[train_ids, :]

train_output = z[train_ids]

# %% Get test data

test_pos = grid[test_ids, :]

test_output = z[test_ids]

# %% Plot training and test data

title_fontsize = 15
axis_fontsize = 11

titles = ['von Mises density', 'Training data', 'Test data']

fig, ax = plt.subplots(1, 3, figsize=[12, 3], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.35,
    hspace=0.0
)

line_width = 2

# https://matplotlib.org/stable/tutorials/colors/colors.html

pdf_line_col = '#069AF3' # azure
circle_line_col = 'black'

train_point_col = '#F97306' # orange
test_point_col = '#C20078' # magenta

# https://matplotlib.org/stable/api/markers_api.html

point_marker = 'o'

point_size = 8

ax[0].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[1].scatter(
    train_pos[:, 0],
    train_pos[:, 1],
    train_output,
    color=train_point_col,
    marker=point_marker,
    s=point_size
)

ax[1].plot(x, y, z, color=pdf_line_col, lw=line_width)

ax[2].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    test_output,
    color=test_point_col,
    marker=point_marker,
    s=point_size
)

ax[2].plot(x, y, z, color=pdf_line_col, lw=line_width)

for i in range(3):
    ax[i].set_proj_type('ortho')

    ax[i].plot(x, y, 0, color=circle_line_col, lw=line_width, zorder=0)

    ax[i].grid(False)

    ax[i].tick_params(pad=-1.5)

    ax[i].set_xlim((-1, 1))
    ax[i].set_ylim((-1, 1))
    ax[i].set_zlim((0, 11))

    ax[i].set_title(titles[i], fontsize=title_fontsize, pad=-1.5)

    ax[i].set_xlabel('x', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_ylabel('y', fontsize=axis_fontsize, labelpad=-3)
    ax[i].set_zlabel('z', fontsize=axis_fontsize, labelpad=-27)

    ax[i].set_xticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_yticks([-1, 0, 1], fontsize=axis_fontsize)
    ax[i].set_zticks([0, 5., 10.], fontsize=axis_fontsize)

    ax[i].zaxis.set_rotate_label(False)

# %% Save plot

plt.savefig(
    output_path.joinpath('data.png'),
    dpi=600,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()
