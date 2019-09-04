import numpy as np
import matplotlib.pyplot as plt

import gaussdca

import settings


def norm(x):
    flat = x.flatten()
    print(0.99 * flat.shape[0], flat.shape[0])
    k = int(round(flat.shape[0] * 0.995))
    return np.partition(flat, k)[k]


data = gaussdca.run('/fat/CASP13/targets/T1001/T1001.jhE0.a3m')

original = data['gdca']
corr = data['gdca_corr']

compare = np.tril(original) / norm(original) + np.triu(corr) / norm(corr)

plt.imshow(compare, cmap='Greys', vmin=-0.1, vmax=2)
plt.ylabel('Original score')
plt.xlabel('With APC')
plt.tight_layout()
plt.savefig('../figures/apc.pdf')
plt.figure()
plt.hist(compare.flatten(), histtype='step', bins='auto')
plt.show()
