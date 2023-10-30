# %%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
r = 10
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z, color='linen', alpha=0.5)

# plot circular curves over the surface
theta = np.linspace(0, 2 * np.pi, 100)
z = np.zeros(100)
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, color='black', alpha=0.75)
ax.plot(z, x, y, color='black', alpha=0.75)

## add axis lines
zeros = np.zeros(1000)
line = np.linspace(-10,10,1000)

ax.plot(line, zeros, zeros, color='black', alpha=0.75)
ax.plot(zeros, line, zeros, color='black', alpha=0.75)
ax.plot(zeros, zeros, line, color='black', alpha=0.75)

plt.show()

# %%
