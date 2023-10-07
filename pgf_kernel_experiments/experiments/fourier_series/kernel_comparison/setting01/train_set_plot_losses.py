# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.fourier_series.kernel_comparison.setting01.set_env import (
    num_runs, output_basepath, output_paths
)

# %% Generate and save plots of losses

all_losses = 0.

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting losses of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

manual_ylim = True
if manual_ylim:
    manual_ylims = [[-2, 2] for _ in range(num_runs)]

labels = ['PGF kernel','RBF kernel', 'Matern kernel', 'Periodic kernel', 'Spectral kernel']
kernel_names = ['pgf', 'rbf', 'matern', 'periodic', 'spectral']

label_fontsize = 11
axis_fontsize = 11
legend_fontsize = 11

for i in range(num_runs):
    # If verbose, state run number

    if verbose:
        print(msg.format(i+1, num_runs))

    # Load losses

    losses = np.loadtxt(
        output_paths[i].joinpath('train_set_losses.csv'),
        delimiter=',',
        skiprows=1
    )

    all_losses = all_losses + losses

    num_iters, num_kernels = losses.shape

    # Plot losses

    plt.figure(figsize=[7, 4])

    plt.margins(0.)

    plt.xlim([1, num_iters+1])
    if manual_ylim:
        plt.ylim(manual_ylims[i])
    else:
        plt.ylim([losses.min(), losses.max()])

    plt.xlabel('Iteration', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)

    handles = []
    for j in range(num_kernels):
        handle, = plt.plot(range(1, num_iters+1), losses[:, j])
        handles.append(handle)

    plt.xticks(np.arange(0, num_iters+50, 50), fontsize=axis_fontsize)
    plt.yticks(np.arange(-2, 2+1, 1), fontsize=axis_fontsize)

    plt.legend(
        handles,
        labels,
        ncol=1,
        frameon=False,
        markerscale=2,
        fontsize=legend_fontsize
    )

    # Save plot

    plt.savefig(
        output_paths[i].joinpath('train_set_losses.pdf'),
        dpi=1200,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()

# %% Generate and save plot of loss means

all_losses = all_losses / num_runs

# Plot loss means

if verbose:
    print('Plotting loss means...')

plt.figure(figsize=[7, 4])

plt.margins(0.)

plt.xlim([1, num_iters+1])
if manual_ylim:
    plt.ylim([-2, 2])
else:
    plt.ylim([all_losses.min(), all_losses.max()])

plt.xlabel('Iteration', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)

handles = []
for j in range(num_kernels):
    handle, = plt.plot(range(1, num_iters+1), all_losses[:, j])
    handles.append(handle)

plt.xticks(np.arange(0, num_iters+50, 50), fontsize=axis_fontsize)
plt.yticks(np.arange(-2, 2+1, 1), fontsize=axis_fontsize)

plt.legend(
    handles,
    labels,
    ncol=1,
    frameon=False,
    markerscale=2,
    fontsize=legend_fontsize
)

# Save plot

plt.savefig(
    output_basepath.joinpath('train_set_loss_means.pdf'),
    dpi=1200,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()
