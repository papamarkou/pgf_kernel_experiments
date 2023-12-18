# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    dpi, num_runs, output_basepath, output_paths
)

# %% Generate and save plots of losses

all_losses = 0.

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting losses of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

manual_ylim = True
if manual_ylim:
    manual_ylims = [[5, 8] for _ in range(num_runs)]

kernel_names = ['pgf', 'rbf', 'matern', 'periodic']
labels = ['PGF kernel','RBF kernel', 'Matern kernel', 'Periodic kernel']

label_fontsize = 11
axis_fontsize = 11
legend_fontsize = 11

for i in range(num_runs):
    # If verbose, state run number

    if verbose:
        print(msg.format(i+1, num_runs))

    # Load losses

    losses = []
    for kernel_name in kernel_names:
        losses.append(np.loadtxt(output_paths[i].joinpath(kernel_name+'_dkl_losses.csv')))
    losses = np.array(losses).transpose()

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
        # handle, = plt.plot(range(1, num_iters+1), np.cumsum(losses[:, j]) / np.arange(1, num_iters+1))
        handles.append(handle)

    plt.xticks(np.arange(0, num_iters+500, 500), fontsize=axis_fontsize)
    if manual_ylim:
        plt.yticks(np.arange(5, 8+1, 1), fontsize=axis_fontsize)

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
        output_paths[i].joinpath('losses.pdf'),
        dpi=dpi,
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
    plt.ylim([5, 8])
else:
    plt.ylim([all_losses.min(), all_losses.max()])

plt.xlabel('Iteration', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)

handles = []
for j in range(num_kernels):
    handle, = plt.plot(range(1, num_iters+1), all_losses[:, j])
    # handle, = plt.plot(range(1, num_iters+1), np.cumsum(all_losses[:, j]) / np.arange(1, num_iters+1))
    handles.append(handle)

plt.xticks(np.arange(0, num_iters+500, 500), fontsize=axis_fontsize)
if manual_ylim:
    plt.yticks(np.arange(5, 8+1, 1), fontsize=axis_fontsize)

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
    output_basepath.joinpath('loss_means.pdf'),
    dpi=dpi,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()
