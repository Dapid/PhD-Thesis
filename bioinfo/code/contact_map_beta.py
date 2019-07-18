import gzip

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

import settings

LOW = 3407
UP = 3443
N =  UP - LOW + 1
print(N)
coords = np.full((N, 3), fill_value=np.nan)
coords_cb = np.full((N, 3), fill_value=np.nan)
for line in gzip.open('/data/pdb_download/data/6D7Y.pdb.gz'):
    line = line.decode()
    if line.startswith('ATOM') or line.startswith('HETAT') and line[21] == 'A' and line[13:15] == 'CA':
        res = int(line[22:26])

        if LOW <= res <= UP:

            pos = res - LOW
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            coords[pos, :] = (x, y, z)
    if line.startswith('ATOM') or line.startswith('HETAT') and line[21] == 'A' and line[13:15] == 'CB':
        res = int(line[22:26])

        if LOW <= res <= UP:

            pos = res - LOW
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            coords_cb[pos, :] = (x, y, z)



dmap = spatial.distance.squareform(spatial.distance.pdist(coords_cb))
cmap = (dmap < 10).astype(np.float) + (dmap < 9).astype(np.float) + (dmap < 8).astype(np.float)
ij = np.triu_indices_from(cmap)
cmap[ij] = np.nan
plt.imshow(cmap, cmap='Greys', alpha=0.8)

plt.xticks([0, 10, 20, 30], [LOW , LOW + 10, LOW + 20, LOW + 30])
plt.yticks([0, 10, 20, 30], [LOW , LOW + 10, LOW + 20, LOW + 30])
plt.xlabel('Residue number')
plt.ylabel('Residue number')
plt.tight_layout()
plt.savefig('../figures/beta_contactmap_base.pdf')
plt.show()