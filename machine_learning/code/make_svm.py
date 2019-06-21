import numpy as np

import matplotlib
from matplotlib import rc

rc('font',**{'family':'serif','serif':['palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage[euler-digits]{eulervm}')

import matplotlib.pyplot as plt
from sklearn import svm, datasets

MAROON='#AD1737'
BLUE = 'RoyalBlue'


iris = datasets.load_iris()
y = iris.target[iris.target < 2]
X = iris.data[iris.target < 2, :2]


clf = svm.LinearSVC(C=5., max_iter=int(1e4))
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(4, 7.2)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')

plt.plot(xx, yy, color='k', alpha=0.5)
plt.plot(xx, yy_down, 'k--', alpha=0.5)
plt.plot(xx, yy_up, 'k--', alpha=0.5)

plt.fill_between(xx, yy, yy_up, alpha=0.2, color=MAROON)
plt.fill_between(xx, yy_down, yy, alpha=0.2, color=BLUE)
plt.fill_between(xx, 1, yy_down, alpha=0.5, color=BLUE)
plt.fill_between(xx, yy_up, 5, alpha=0.5, color=MAROON)


plt.xlim(xx.min(), xx.max())
plt.ylim(1, 5)

plt.legend(loc=0)

plt.title('Linear SVM')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.tight_layout()

plt.savefig('../figures/svm.pdf')
plt.show()
