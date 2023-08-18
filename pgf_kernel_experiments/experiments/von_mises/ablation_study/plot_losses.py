# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.von_mises.ablation_study.set_env import output_path

# %% Load losses

losses = np.loadtxt(
    output_path.joinpath('losses.csv'),
    delimiter=',',
    skiprows=1
)

num_iters, _ = losses.shape

width_idx = [0, 1, 2]
depth_idx = [3, 4, 5]

# %% Configure plotting setup

width_labels = [r'$m_1 = 2$', r'$m_1 = 100$', r'$m_1 = 200$']
depth_labels = [r'$m_1 = 10$', r'$m_1 = 10, m_2 = 10$', r'$m_1 = 10, m_2 = 10, m_3 = 10$']

width_fname = 'width_losses.png'
depth_fname = 'depth_losses.png'

label_fontsize = 11
axis_fontsize = 11
legend_fontsize = 11

# %% Plot and save losses

for idx, labels, fname in zip([width_idx, depth_idx], [width_labels, depth_labels], [width_fname, depth_fname]):
    plt.figure(figsize=[7, 4])

    plt.margins(0.)

    plt.xlabel('Iteration', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)

    plt.xlim([1, num_iters+1])
    plt.ylim([2., 10])

    handles = []
    for i in idx:
        handle, = plt.plot(range(1, num_iters+1), losses[:, i])
        handles.append(handle)

    plt.xticks(np.arange(0, num_iters+50, 50), fontsize=axis_fontsize)
    plt.yticks(np.arange(2, 10+2, 2), fontsize=axis_fontsize)

    plt.legend(
        handles,
        labels,
        ncol=1,
        frameon=False,
        markerscale=2,
        fontsize=legend_fontsize
    )

    plt.savefig(
        output_path.joinpath(fname),
        dpi=600,
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()
