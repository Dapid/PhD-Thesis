import settings

from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt

import tables
import numpy as np

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'${\text{-}%s}$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\nicefrac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'${\text{-}\nicefrac{%s}{%s}}$'%(latex,den)
            else:
                return r'$\nicefrac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))



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

f, ax = plt.subplots()
dh = np.concatenate(dh)
rsa = np.concatenate(rsa)
rsa[rsa.argmax()] = 1
print(rsa.min())
plt.scatter(dh[:, 0], dh[:, 1], c=rsa, alpha=0.2)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$\psi$')

plt.colorbar(label='RSA')
ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))

plt.tight_layout()
plt.savefig('../figures/transfer_learning.pdf')
plt.show()
