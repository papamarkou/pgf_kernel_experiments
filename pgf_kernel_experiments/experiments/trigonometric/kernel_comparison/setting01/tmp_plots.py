# %%

import matplotlib.pyplot as plt
import numpy as np

# %%

losses = []

losses.append(np.loadtxt(("output/run1/spectral_gp_losses.csv"), skiprows=1))

losses.append(np.loadtxt(("output/run1/periodic_gp_losses.csv"), skiprows=1))

losses.append(np.loadtxt(("output/run1/matern_gp_losses.csv"), skiprows=1))

losses.append(np.loadtxt(("output/run1/rbf_gp_losses.csv"), skiprows=1))

losses.append(np.loadtxt(("output/run1/pgf_gp_losses.csv"), skiprows=1))

# %%

loss_run_mean = []

for i in range(5):
    loss_run_mean.append(np.cumsum(losses[i]) / np.arange(1, losses[i].shape[0]+1))

# %%

for i in range(5):
    plt.plot(losses[i])

# %%

for i in range(5):
    plt.plot(loss_run_mean[i])

# %%
