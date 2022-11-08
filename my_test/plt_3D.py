import torch
import numpy as np
import matplotlib.pyplot as plt

# define the axes
fig = plt.figure(0)
ax = plt.axes(projection='3d')


xx = np.linspace(-30, 90, 5)
yy = np.linspace(-30, 90, 3)

X, Y = np.meshgrid(xx, yy, indexing="ij")
X_t, Y_t = torch.meshgrid(torch.tensor(xx), torch.tensor(yy), indexing="xy")
# the numpy
print(f"X's shape: {X.shape}")
print(f"Y's shape: {Y.shape}")
# the torch
print(f"X_t's shape: {X_t.shape}")
print(f"Y_t's shape: {Y_t.shape}")
# cout the X and the Y
print(X)
print(Y)
#Z = np.power((2*X + 3*Y - 100), 2)
##ax.plot_surface(X, Y, Z, cmap='rainbow')
#ax.contour(X, Y, Z, zdir='z', offset=-6, cmap='rainbow')
#
#plt.show()
#print(f"xx shape: {xx.shape}")
#print(f"X shape: {X.shape}")
#print(f"Y shape: {Y.shape}")