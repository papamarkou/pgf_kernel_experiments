# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting02.set_env import data_path, output_path

# %% Load data

data = np.loadtxt(
    data_path.joinpath('data.csv'),
    delimiter=',',
    skiprows=1
)

grid = data[:, 1:3]
x = data[:, 1]
y = data[:, 2]
z_signal = data[:, 3]
z = data[:, 5]

train_ids = np.loadtxt(data_path.joinpath('train_ids.csv'), dtype='int')

test_ids = np.loadtxt(data_path.joinpath('test_ids.csv'), dtype='int')

# %% Get training data

train_pos = grid[train_ids, :]
train_output = z[train_ids]

# %% Get test data

test_pos = grid[test_ids, :]
test_output = z[test_ids]

# %% Load predictions

predictions = np.loadtxt(
    output_path.joinpath('predictions.csv'),
    delimiter=',',
    skiprows=1
)

# %% Plot predictions

title_fontsize = 15
axis_fontsize = 11

titles = [
    ['von Mises density', 'Training data', 'Test data', 'PGF kernel'],
    ['RBF kernel', 'Matern kernel', 'Periodic kernel', 'Spectral kernel']
]

fig, ax = plt.subplots(2, 4, figsize=[14, 6], subplot_kw={'projection': '3d'})

fig.subplots_adjust(
    left=0.0,
    bottom=0.0,
    right=1.0,
    top=1.0,
    wspace=-0.4,
    hspace=0.15
)

line_width = 2

# https://matplotlib.org/stable/tutorials/colors/colors.html

pdf_line_col = '#069AF3' # azure
circle_line_col = 'black'

train_point_col = '#F97306' # orange
test_point_col = '#C20078' # magenta
pred_point_col = '#E50000' # red

# https://matplotlib.org/stable/api/markers_api.html

point_marker = 'o'

point_size = 8

ax[0, 0].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[0, 1].scatter(
    train_pos[:, 0],
    train_pos[:, 1],
    train_output,
    color=train_point_col,
    marker=point_marker,
    s=point_size
)

ax[0, 1].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[0, 2].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    test_output,
    color=test_point_col,
    marker=point_marker,
    s=point_size
)

ax[0, 2].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[0, 3].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions[:, 0],
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[0, 3].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[1, 0].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions[:, 1],
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[1, 0].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[1, 1].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions[:, 2],
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[1, 1].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[1, 2].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions[:, 3],
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[1, 2].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

ax[1, 3].scatter(
    test_pos[:, 0],
    test_pos[:, 1],
    predictions[:, 4],
    color=pred_point_col,
    marker=point_marker,
    s=point_size
)

ax[1, 3].plot(x, y, z_signal, color=pdf_line_col, lw=line_width)

for i in range(2):
    for j in range(4):
        ax[i, j].set_proj_type('ortho')

        ax[i, j].plot(x, y, 0, color=circle_line_col, lw=line_width, zorder=0)

        ax[i, j].grid(False)

        ax[i, j].tick_params(pad=-1.5)
        
        ax[i, j].set_xlim((-1, 1))
        ax[i, j].set_ylim((-1, 1))
        ax[i, j].set_zlim((-2, 12))

        ax[i, j].set_title(titles[i][j], fontsize=title_fontsize, pad=-1.5)

        ax[i, j].set_xlabel('x', fontsize=axis_fontsize, labelpad=-3)
        ax[i, j].set_ylabel('y', fontsize=axis_fontsize, labelpad=-3)
        ax[i, j].set_zlabel('z', fontsize=axis_fontsize, labelpad=-27)

        ax[i, j].set_xticks([-1, 0, 1], fontsize=axis_fontsize)
        ax[i, j].set_yticks([-1, 0, 1], fontsize=axis_fontsize)
        ax[i, j].set_zticks([-2, 5., 12.], fontsize=axis_fontsize)

        ax[i, j].zaxis.set_rotate_label(False)

# %% Save plot

plt.savefig(
    output_path.joinpath('predictions.png'),
    dpi=600,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()
