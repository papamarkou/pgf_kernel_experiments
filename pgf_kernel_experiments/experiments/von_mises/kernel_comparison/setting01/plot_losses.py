# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting01.set_env import num_runs, output_paths

# %% Generate and save plots of data

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting losses of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}'

manual_ylim = True
if manual_ylim:
    manual_ylims = [[-2, 10] for _ in range(num_runs)]

labels = ['PGF kernel','RBF kernel', 'Matern kernel', 'Periodic kernel', 'Spectral kernel']

label_fontsize = 11
axis_fontsize = 11
legend_fontsize = 11

for i in range(num_runs):
    # If verbose, state run number

    if verbose:
        print(msg.format(i+1, num_runs))

    # Load losses

    losses = np.loadtxt(
        output_paths[i].joinpath('losses.csv'),
        delimiter=',',
        skiprows=1
    )

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
    plt.yticks(np.arange(-2, 10+2, 2), fontsize=axis_fontsize)

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
        output_paths[i].joinpath('losses.png'),
        dpi=600,
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()
