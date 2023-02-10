# %%

from qutip import Bloch
from math import sqrt, sin, cos, pi
from colorsys import hsv_to_rgb

from scipy.interpolate import interp2d
from numpy.random import rand

from numpy import linspace, outer, ones, sin, cos, arccos, arctan2, size, empty

import scipy

import matplotlib.pyplot as plt

import numpy as np

# %%

class BlochDensity(Bloch):
  def __init__(self, fig=None, axes=None, view=None, figsize=None, background=False):
    super.__init__(fig=fig, axes=axes, view=view, figsize=figsize, background=background)

  def plot_back(self):
    # back half of sphere
    u = linspace(0, pi, 25)
    v = linspace(0, pi, 25)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(size(u)), cos(v))

    colours = empty(x.shape, dtype=object)

    for i in range(len(x)):
      for j in range(len(y)):
        theta = arctan2(y[i,j], x[i,j])
        phi = arccos(z[i,j])

        colours[i,j] = self.density(theta, phi)


    self.axes.plot_surface(x, y, z, rstride=1, cstride=1,
                           facecolors=colours,
                           alpha=self.sphere_alpha, 
                           linewidth=0, antialiased=True)
    # wireframe
    self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                             color=self.frame_color,
                             alpha=self.frame_alpha)
    # equator
    self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z',
                   lw=self.frame_width, color=self.frame_color)
    self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x',
                   lw=self.frame_width, color=self.frame_color)

  def plot_front(self):
    # front half of sphere
    u = linspace(-pi, 0, 25)
    v = linspace(0, pi, 25)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(size(u)), cos(v))

    colours = empty(x.shape, dtype=object)
    for i in range(len(x)):
      for j in range(len(y)):
        theta = arctan2(y[i,j], x[i,j])
        phi = arccos(z[i,j])

        colours[i,j] = self.density(theta, phi)

    self.axes.plot_surface(x, y, z, rstride=1, cstride=1,
                           facecolors=colours,
                           alpha=self.sphere_alpha, 
                           linewidth=0, antialiased=True)


    # wireframe
    self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                             color=self.frame_color,
                             alpha=self.frame_alpha)
    # equator
    self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                   zs=0, zdir='z', lw=self.frame_width,
                   color=self.frame_color)
    self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                   zs=0, zdir='x', lw=self.frame_width,
                   color=self.frame_color)

# %%

b = BlochDensity()
b.sphere_alpha=0.5

thetas, phis = linspace(-pi,pi, 20), linspace(0,pi, 20)
density = rand(len(thetas), len(phis))

#scale density to a maximum of 1
# density /= density.max()

# interpolated_density = interp2d(thetas, phis, density)

interpolated_density = scipy.interpolate.RegularGridInterpolator((thetas, phis), density)

def f(theta, phi):
  # return hsv_to_rgb(interpolated_density(theta,phi)[0, 0], 1, 1)
  # return interpolated_density(theta,phi)[0, 0]
  return hsv_to_rgb(interpolated_density(np.array([theta, phi]))[0], 1, 1)

b.density = f

b.show()

# %%

u = linspace(0, pi, 25)
v = linspace(0, pi, 25)
x = outer(cos(u), sin(v))
y = outer(sin(u), sin(v))
z = outer(ones(size(u)), cos(v))

colours = empty(x.shape, dtype=object)

for i in range(len(x)):
    for j in range(len(y)):
        theta = arctan2(y[i,j], x[i,j])
        phi = arccos(z[i,j])

        colours[i,j] = b.density(theta, phi)

# %%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# fig = plt.figure()
# ax = fig.add_subplot(aspect='equal', projection='3d')

# ax.set_proj_type('ortho')

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.set_aspect('equal','box')

ax.set_box_aspect((1, 1, 1))

ax.grid(False)

# ax.view_init(30, 120)

ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           facecolors=colours,
                           alpha=b.sphere_alpha, 
                           linewidth=0, antialiased=True)


ax.plot(1.0 * cos(u), 1.0 * sin(u),
                   zs=0, zdir='z', lw=2,
                   color='black')
ax.plot(1.0 * cos(u), 1.0 * sin(u),
                   zs=0, zdir='x', lw=2,
                   color='black')

ax.zaxis.set_rotate_label(False)

# %%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(constrained_layout=True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(10), range(10), "o-")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
b1 = Bloch(fig=fig, axes=ax2)
b1.render()
ax2.set_box_aspect([1, 1, 1]) # required for mpl > 3.1

plt.show()

# %%
