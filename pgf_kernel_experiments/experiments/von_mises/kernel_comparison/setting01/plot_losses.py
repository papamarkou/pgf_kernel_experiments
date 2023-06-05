# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.von_mises.kernel_comparison.setting01.set_env import output_path

# %% Load predictions

losses = np.loadtxt(
    output_path.joinpath('losses.csv'),
    delimiter=',',
    skiprows=1
)

num_iters, num_kernels = losses.shape

# %% Plot losses

plt.figure(figsize=[7, 4])

plt.margins(0.)

plt.xlim([1, num_iters+1])
plt.ylim([losses.min(), losses.max()])

for i in range(num_kernels):
    plt.plot(range(1, num_iters+1), losses[:, i])

# %%
