# %%

import matplotlib.pyplot as plt
import numpy as np

# %%

losses = np.loadtxt(("output/run1/spectral_gp_losses.csv"), skiprows=1)

loss_run_mean = np.cumsum(losses) / np.arange(losses.shape[0])

# %%

plt.plot(losses)

plt.plot(loss_run_mean)

# %%
