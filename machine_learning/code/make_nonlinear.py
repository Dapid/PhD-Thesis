import numpy as np

import matplotlib.pyplot as plt
from settings import MAROON, BLUE

x = np.linspace(-5, 5, num=1000)


def line():
    plt.axhline(0, color='k', alpha=0.5)
    plt.axvline(0, color='k', alpha=0.5)


NON = BLUE
plt.plot(x, 1 / (1 + np.exp(-x)), color=MAROON)
line()

plt.title('Sigmoid')
plt.figure()
plt.plot(x, np.tanh(x), color=MAROON)
line()

plt.title('Hyperbolic tangent')
plt.figure()
plt.plot(x, np.where(x > 0, x, 0), color=NON)
line()
plt.title('ReLU')
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig('../figures/relu.pdf')
plt.close()
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig('../figures/tanh.pdf')
plt.close()
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig('../figures/sigmoid.pdf')

plt.figure()
plt.plot(x, np.where(x > 0, x, np.exp(x) - 1), color=NON)
line()

plt.title('Exponential Linear Unit')
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig('../figures/elu.pdf')
