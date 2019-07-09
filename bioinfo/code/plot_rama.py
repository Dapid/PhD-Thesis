import settings
import matplotlib.pyplot as plt

import numpy as np
import tables

h5 = tables.open_file('/home/david/research/PQ4/ss_pred/data/data_jhE3.ur50.v2.h5')

dh = []
rsa = []
for g in h5.root:
    if not hasattr(g, 'valid'): continue

    valid = g.valid[:].squeeze()
    valid = np.convolve(valid, [1, 1, 1], mode='same') >= 3
    dh.append(g.dihedrals[:].squeeze()[valid])
    rsa.append(np.clip(g.rsa[:].squeeze()[valid], 0, 1))

    if len(rsa) > 10:
        break

h5.close()

plt.figure()
dh = np.concatenate(dh)
rsa = np.concatenate(rsa)
rsa[rsa.argmax()] = 1
print(rsa.min())
plt.scatter(dh[:, 0], dh[:, 1], c=rsa, alpha=0.2)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$\psi$')

plt.colorbar(label='RSA')

plt.tight_layout()
plt.show()
plt.savefig('../figures/transfer_learning.pdf')
