import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import settings

f_c = np.load('combined.npz')
f_s = np.load('separated.npz')

plt.figure()
plt.scatter(f_c['phi_pred'], f_c['psi_pred'])

plt.figure()
plt.scatter(f_s['phi_pred'], f_s['psi_pred'])

plt.show()