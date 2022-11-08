import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1e-3, 1-1e-3, 1000)

y = -x*np.log2(x) -(1-x)*np.log2(1-x)

# plot the figure
plt.figure(0)
plt.title("y=-x*log2(x)-(1-x)log2(1-x)")

plt.xlabel("x")
plt.ylabel("y")

plt.plot(x, y)
plt.savefig("../save_figs/y=-x*log2(x)-(1-x)log2(1-x).jpg")
plt.show()