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
