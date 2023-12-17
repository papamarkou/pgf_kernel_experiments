# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, dpi, num_classes, num_runs, output_paths
)

# %% Generate and save plots of predictions with periodic kernel

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting predictions of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

xyz_lim = 0.63

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

    projected_x = np.loadtxt(output_paths[i].joinpath('periodic_dkl_projections.csv'), delimiter=',')

    # Set up figure

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Plot spherical surface

    ax.plot_surface(
        surface['x'],
        surface['y'],
        surface['z'],
        cstride=1,
        rstride=1,
        edgecolor='none',
        alpha=0.15
    )

    # Plot projections

    for j in range(num_classes):
        ax.scatter(
            projected_x[labels == j, 0],
            projected_x[labels == j, 1],
            projected_x[labels == j, 2],
            color='C'+str(j),
            s=10
        )

    # Configure plot

    ax.set_box_aspect([1, 1, 1])

    ax.set_aspect("equal")

    ax.grid(False)
    ax.axis('off')

    ax.set_xlim(-xyz_lim, xyz_lim)
    ax.set_ylim(-xyz_lim, xyz_lim)
    ax.set_zlim(-xyz_lim, xyz_lim)

    # Save plot

    plt.savefig(
        output_paths[i].joinpath('periodic_dkl_predictions.pdf'),
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close(fig)
