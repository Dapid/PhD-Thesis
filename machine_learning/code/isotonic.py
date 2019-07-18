import os
import time

import yaml
import numpy as np
from scipy import stats

import settings
import pylab as plt
from sklearn.isotonic import IsotonicRegression

import tqdm

base_dir = '/fat/research/retrain_pconsc4/data'

test = yaml.unsafe_load(open(os.path.join(base_dir, 'test_contact_index.yaml')))

gdca_scores = []
is_contact = []

count = 0

for pdb, alignmens in tqdm.tqdm(test.values()):
    if any('jhE0' in ali for ali in alignmens):
        ali = [ali for ali in alignmens if 'jhE0' in ali][0]
    else:
        continue

    try:
        gdca = np.load(ali)['gdca'].squeeze()
        dmap = np.load(pdb)['dmap']
    except FileNotFoundError:
        continue
    valid = np.isfinite(dmap)
    cmap = dmap < 8
    N = gdca.shape[0]
    for i in range(N):
        for j in range(i + 5, N):
            if valid[i, j]:
                gdca_scores.append(gdca[i, j])
                is_contact.append(cmap[i, j])
    count += 1

    if count > 500:
        break

t0 = time.time()
gdca_scores = np.array(gdca_scores, dtype=np.float32)
is_contact = np.array(is_contact, dtype=np.float32)

iso = IsotonicRegression()
iso.fit(gdca_scores, is_contact)
print(time.time() - t0)

x = np.linspace(min(gdca_scores), max(gdca_scores), num=1000)


bins = 200

SIZE = 2
sc_0 = plt.scatter(gdca_scores[is_contact < 0.5], is_contact[is_contact < 0.5], alpha=0.1, s=SIZE, color='k')
sc_1 = plt.scatter(gdca_scores[is_contact > 0.5], is_contact[is_contact > 0.5], alpha=0.1, s=SIZE, color='k')
sc_0.set_rasterized(True)
sc_1.set_rasterized(True)
print('--', time.time() - t0)

mean, edges, _ = stats.binned_statistic(gdca_scores, is_contact, bins=bins)
centres = (edges[:-1] + edges[1:]) / 2
plt.plot(centres, mean, color=settings.BLUE, alpha=0.5, linestyle='--')
plt.plot(x, iso.transform(x), color=settings.MAROON, linewidth=2)


plt.xlabel('Contact score')
plt.ylabel('Contact probability')
#plt.title('Isotonic regression')

plt.xlim(-1.2, 4.2)
plt.xticks(range(-1, 5))
plt.tight_layout()

print('.', time.time() - t0)
plt.savefig('../figures/isotonic.pdf')
print('!', time.time() - t0)
plt.show()
