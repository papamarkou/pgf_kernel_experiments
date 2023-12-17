# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, dpi, num_classes, num_runs, output_paths
)
from pgf_kernel_experiments.plots import set_axes_equal

# %% Generate and save plots of projections with PGF kernel

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting predictions of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

xyz_lim = 0.63

title_fontsize = 15
colorbar_fontsize = 11

kernel_keys = {
    0 : 'pgf',
    1 : 'rbf',
    2 : 'matern',
    3 : 'periodic',
    4 : 'spectral' 
}

titles = {
    'pgf' : 'PGF kernel',
    'rbf' : 'RBF kernel',
    'matern' : 'Matern kernel',
    'periodic' : 'Periodic kernel',
    'spectral' : 'Spectral kernel'
}

# %% Generate data that determine the spherical surface

# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere

surface = {'r' : 1.}
surface['phi'], surface['theta'] = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
surface['x'] = surface['r'] * np.sin(surface['phi']) * np.cos(surface['theta'])
surface['y'] = surface['r'] * np.sin(surface['phi']) * np.sin(surface['theta'])
surface['z'] = surface['r'] * np.cos(surface['phi'])

# %% Generate and save plots of projections

# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere

for i in range(num_runs):
    # If verbose, state run number

    if verbose:
        print(msg.format(i+1, num_runs))

    # Load labels

    labels = np.loadtxt(data_paths[i].joinpath('labels.csv'), dtype='int')
    train_ids = np.loadtxt(data_paths[i].joinpath('train_ids.csv'), dtype='int')
    labels = labels[train_ids]

    # Load projections

    projected_x = {}
    projected_x['pgf'] = np.loadtxt(output_paths[i].joinpath('pgf_dkl_projections.csv'), delimiter=',')
    projected_x['rbf'] = np.loadtxt(output_paths[i].joinpath('rbf_dkl_projections.csv'), delimiter=',')
    projected_x['matern'] = np.loadtxt(output_paths[i].joinpath('matern_dkl_projections.csv'), delimiter=',')
    projected_x['periodic'] = np.loadtxt(output_paths[i].joinpath('periodic_dkl_projections.csv'), delimiter=',')
    projected_x['spectral'] = np.loadtxt(output_paths[i].joinpath('spectral_dkl_projections.csv'), delimiter=',')

    # Plot data, including separate training and test data (adding color map)

    fig = plt.figure(figsize=[14, 6])

    fig.subplots_adjust(
        left=0.0,
        bottom=0.0,
        right=1.0,
        top=1.0,
        wspace=-0.64,
        hspace=0.15
    )

    for j in range(len(kernel_keys)):
        ax = fig.add_subplot(2, 3, j+1, projection='3d')

        ax.plot_surface(
            surface['x'],
            surface['y'],
            surface['z'],
            cstride=1,
            rstride=1,
            edgecolor='none',
            alpha=0.15
        )

        for k in range(num_classes):
            ax.scatter(
                projected_x[kernel_keys[j]][labels == k, 0],
                projected_x[kernel_keys[j]][labels == k, 1],
                projected_x[kernel_keys[j]][labels == k, 2],
                color='C'+str(k),
                s=10
            )

        ax.set_title(titles[kernel_keys[j]], fontsize=title_fontsize)

        ax.set_box_aspect([1, 1, 1])

        set_axes_equal(ax)

        ax.grid(False)
        ax.axis('off')

        ax.set_xlim(-xyz_lim, xyz_lim)
        ax.set_ylim(-xyz_lim, xyz_lim)
        ax.set_zlim(-xyz_lim, xyz_lim)

    # Save plot

    plt.savefig(
        output_paths[i].joinpath('predictions.pdf'),
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close(fig)
