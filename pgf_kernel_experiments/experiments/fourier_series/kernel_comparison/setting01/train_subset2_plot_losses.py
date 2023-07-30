# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.fourier_series.kernel_comparison.setting01.set_env import output_path

# %% Load predictions

losses = np.loadtxt(
    output_path.joinpath('train_subset2_losses.csv'),
    delimiter=',',
    skiprows=1
)

num_iters, num_kernels = losses.shape

# %% Plot losses

labels = ['PGF kernel','RBF kernel', 'Matern kernel', 'Periodic kernel', 'Spectral kernel']

label_fontsize = 11
axis_fontsize = 11
legend_fontsize = 11

plt.figure(figsize=[7, 4])

plt.margins(0.)

plt.xlabel('Iteration', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)

plt.xlim([1, num_iters+1])
# plt.ylim([losses.min(), losses.max()])
plt.ylim([-2., 2.])

handles = []
for i in range(num_kernels):
    handle, = plt.plot(range(1, num_iters+1), losses[:, i])
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

# %% Save plot

plt.savefig(
    output_path.joinpath('train_subset2_losses.png'),
    dpi=600,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.close()
