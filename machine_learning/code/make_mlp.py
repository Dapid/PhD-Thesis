import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets, neural_network

from settings import MAROON, BLUE

iris = datasets.load_iris()
y = iris.target[iris.target < 2]
X = iris.data[iris.target < 2, :2]

clf = neural_network.MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=1000, activation='relu')
clf.fit(X, y)

xx = np.linspace(4, 7.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')

plt.xlim(xx.min(), xx.max())
plt.ylim(1, 5)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
h = 0.002
xx, yy = np.meshgrid(np.arange(xx.min(), xx.max() + h, h),
                     np.arange(1, 5 + h, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 0] - Z[:, 1]
Z = Z.reshape(xx.shape)

ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.contourf(xx, yy, Z, colors=[MAROON, BLUE], levels=[-1, 0, 1], alpha=0.2)
ax.contourf(xx, yy, Z, colors=[MAROON, ], levels=[-100, -1], alpha=0.5)
ax.contourf(xx, yy, Z, colors=[BLUE, ], levels=[1, 1000], alpha=0.5)
plt.legend(loc=1)

plt.title('MLP')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.tight_layout()

plt.figure()
plt.subplot(121)
plt.hist(Z.flatten(), histtype='step')
plt.subplot(122)
plt.imshow(Z)
plt.show()
