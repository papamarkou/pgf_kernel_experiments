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

ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           facecolors=colours,
                           alpha=b.sphere_alpha, 
                           linewidth=0, antialiased=True)

# %%
