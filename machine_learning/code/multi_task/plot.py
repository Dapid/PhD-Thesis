import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import settings

f_c = np.load('combined.npz')
f_s = np.load('separated.npz')

opts = dict(norm=LogNorm(), bins=100)

plt.figure()
plt.title('Joint')
plt.hist2d(f_c['phi_pred'], f_c['psi_pred'], **opts)

plt.figure()
plt.title('Separated')
plt.hist2d(f_s['phi_pred'], f_s['psi_pred'], **opts)

plt.show()