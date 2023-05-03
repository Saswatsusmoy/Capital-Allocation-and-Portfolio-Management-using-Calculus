import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(W1, W2):
    return W1*26.412 + W2*32.504/np.sqrt((W1*27.59)**2 + (W2*15.002)**2)

W1, W2 = np.linspace(0, 1, num=100), np.linspace(0, 1, num=100)
W1, W2 = np.meshgrid(W1, W2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, f(W1, W2))
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('f(W1, W2)')
plt.show()
