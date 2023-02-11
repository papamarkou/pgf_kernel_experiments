# %% Import packages

import numpy as np

from colorsys import hsv_to_rgb
from qutip import Bloch

# %% BlockDensity class

class BlochDensity(Bloch):
    def __init__(
        self,
        phi_front, theta_front, x_front, y_front, z_front, freqs_front,
        phi_back, theta_back, x_back, y_back, z_back, freqs_back,
        alpha=0.5, fig=None, axes=None, view=None, figsize=None, background=False
    ):
        super.__init__(fig=fig, axes=axes, view=view, figsize=figsize, background=background)

        self.phi_front = phi_front
        self.theta_front = theta_front
        self.x_front = x_front
        self.y_front = y_front
        self.z_front = z_front
        self.freqs_front = freqs_front

        self.phi_back = phi_back
        self.theta_back = theta_back
        self.x_back = x_back
        self.y_back = y_back
        self.z_back = z_back
        self.freqs_back = freqs_back

        self.alpha = alpha

    def set_colours(self, freqs):
        colours = np.empty(freqs.shape, dtype=object)

        n_rows, n_cols = freqs.shape

        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(freqs[i, j]):
                    colours[i, j] = (1., 1., 1.)
                else:
                    colours[i, j] = hsv_to_rgb(freqs[i, j], 1., 1.)

        return colours

    def plot_back(self):
        colours = self.set_colours(self.freqs_front)

        self.axes.plot_surface(
            self.x_back, self.y_back, self.z_back,
            rstride=1, cstride=1, facecolors=colours, alpha=self.alpha, linewidth=0, antialiased=True
        )
