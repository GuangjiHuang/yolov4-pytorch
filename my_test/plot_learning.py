import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)

y1 = np.where(np.abs(x)<=1, 0.5 * x**2, 0)
# the different label
y1 = np.where(np.abs(x)>1, np.abs(x)-0.5, y1)
y2 = np.abs(x)
y3 = x ** 2

plt.figure(1)
plt.title("test")
# plot the label
plt.xlabel("x")
plt.ylabel("loss")

# plot the cure
plt.plot(x, y1, label="smooth l1")
plt.plot(x, y2, label="L1 loss")
plt.plot(x, y3, label="L2 loss")

# show the legend
plt.legend()

# show the figure
plt.show()

