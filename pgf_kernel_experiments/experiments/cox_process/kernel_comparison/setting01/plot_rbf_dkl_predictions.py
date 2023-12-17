# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, num_runs, output_paths
)

# %% Generate and save plots of predictions with RBF kernel

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting predictions of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

# title_fontsize = 15
axis_fontsize = 11

# %%

i = 0

# Load labels

labels = np.loadtxt(data_paths[i].joinpath('labels.csv'), dtype='int')
train_ids = np.loadtxt(data_paths[i].joinpath('train_ids.csv'), dtype='int')
labels = labels[train_ids]

# Load projections

projected_x = np.loadtxt(output_paths[i].joinpath('rbf_dkl_projections.csv'), delimiter=',')

# %% Generate data that determine the spherical surface

# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere

surface = {'r' : 1.}
surface['phi'], surface['theta'] = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
surface['x'] = surface['r'] * np.sin(surface['phi']) * np.cos(surface['theta'])
surface['y'] = surface['r'] * np.sin(surface['phi']) * np.sin(surface['theta'])
surface['z'] = surface['r'] * np.cos(surface['phi'])

# %% Plot projections

# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    surface['x'],
    surface['y'],
    surface['z'],
    rstride=1,
    cstride=1,
    color='c',
    alpha=0.3,
    linewidth=0
)

ax.scatter(
    projected_x[:, 0],
    projected_x[:, 1],
    projected_x[:, 2],
    color="k",
    s=20
)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")

# %%
