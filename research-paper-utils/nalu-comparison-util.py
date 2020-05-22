import scipy
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
# Changing figure's font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"



'''
NAC 3-D Plot Code
'''
# 3-D axis creation
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')


# Two functions for multiplication, and creating transformation matrix W
sigmoid = lambda y: 1 / (1 + np.exp(-y))
tanh = lambda x: ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

# Make data
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
# Final function for Transformation Matrix W
Z = tanh(X)*sigmoid(Y)

# Axis Labelling
ax.set_xlabel('tanh(x)', fontsize=14)
ax.set_ylabel('sigmoid(y)', fontsize=14)
ax.set_zlabel('W', fontsize=16)
ax.set_title('(a) NAC surface plot', fontsize=18, y=-0.04)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.11, 1.11)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, aspect=6)


'''
NALU 3-D Plot Code
'''

ax = fig.add_subplot(1, 2, 2, projection='3d')

# Sub-function decleration for NALU units
sigmoid = lambda y: 1 / (1 + np.exp(-y))
tanh = lambda x: ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
# learned gate
g_x = lambda x: 1 / (1 + np.exp(-x))
# small epsilon value
eps = 10e-2
# log-exponential space
m_y = lambda y: np.exp(np.log(np.abs(y)+eps))

# Make data
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)

# Final function for Transformation Matrix W
W_nac = tanh(X)*sigmoid(Y)
# Final function for NALU
Z = ( (g_x(Y)*W_nac) + ((1-g_x(Y))*m_y(Y)) )

# Axis Labelling
ax.set_xlabel('tanh(x)', fontsize=14)
ax.set_ylabel('sig-M/G(y), exp(log(|y|+eps))', fontsize=14)
ax.set_zlabel('Y', fontsize=16)
ax.set_title('(b) NALU surface plot',  fontsize=18, y=-0.04)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.1, 10.1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.4, aspect=6)

# show plot
plt.show()
# save plot in given directory
fig.savefig('final_assets/nac-nalu-comparison.png')
