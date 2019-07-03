import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

from settings import MAROON, BLUE

iris = datasets.load_iris()
y = iris.target[iris.target < 2]
X = iris.data[iris.target < 2, :2]

for exp in (3, 1):
    plt.figure()
    clf = linear_model.LogisticRegression(C=10**exp)
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
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    ax.contourf(XX, YY, Z, colors=[MAROON, BLUE], levels=[-1, 0, 1], alpha=0.2)
    ax.contourf(XX, YY, Z, colors=[MAROON, ], levels=[-100, -1], alpha=0.5)
    ax.contourf(XX, YY, Z, colors=[BLUE, ], levels=[1, 1000], alpha=0.5)
    plt.legend(loc=1)

    plt.title('Logistic regression, $L^2=10^{-' + f'{exp}' + '}$')

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.tight_layout()

    plt.savefig(f'../figures/logistic_{exp}.pdf')
plt.show()
