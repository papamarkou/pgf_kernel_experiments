# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, num_runs, output_paths
)

# %% Generate and save plots of predictions with RBF kernel

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting predictions of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

# title_fontsize = 15
axis_fontsize = 11

# %%

i = 0

# Load labels

labels = np.loadtxt(data_paths[i].joinpath('labels.csv'), dtype='int')
train_ids = np.loadtxt(data_paths[i].joinpath('train_ids.csv'), dtype='int')
labels = labels[train_ids]

# Load projections

projected_x = np.loadtxt(output_paths[i].joinpath('rbf_dkl_projections.csv'), delimiter=',')

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    projected_x[:, 0],
    projected_x[:, 1],
    projected_x[:, 2],
    rstride=1,
    cstride=1,
    color='c',
    alpha=0.3,
    linewidth=0
)

# %%

# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %%

# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

# #Import data
# data = np.genfromtxt('leb.txt')
# theta, phi, r = np.hsplit(data, 3) 
# theta = theta * pi / 180.0
# phi = phi * pi / 180.0
# xx = sin(phi)*cos(theta)
# yy = sin(phi)*sin(theta)
# zz = cos(phi)

xx = projected_x[:, 0]
yy = projected_x[:, 1]
zz = projected_x[:, 2]

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

ax.scatter(xx,yy,zz,color="k",s=20)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")

# %%
