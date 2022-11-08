import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-2, 4, 500)

y1 = np.exp(-x)

y2 = np.log(1+np.exp(-x))

y3 = np.where(1-x>=0, 1-x, 0)

y4 = np.where(x>=0, 0, 1)

# plot the figure

plt.figure(0)
plt.title("classification loss function")

# the label
plt.xlabel("fy")
plt.ylabel("loss")

plt.plot(x, y1, label="exp(-fy)")
plt.plot(x, y2, label="log(1+exp(-fy)")
plt.plot(x, y3, label="max{0, 1-fy")
plt.plot(x, y4, label="0-1 loss")

plt.legend()

plt.savefig("../save_figs/loss_function.jpg")
plt.show()

