import numpy as np

import matplotlib
from matplotlib import rc

rc('font',**{'family':'serif','serif':['palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage[euler-digits]{eulervm}')

import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

MAROON='#AD1737'
BLUE = 'RoyalBlue'
GREEN = (0, 0.5, 0)


iris = datasets.load_iris()
y = iris.target[:]
X = iris.data[:, :2]


k=1
for k in (1,  5, 10):
#for ix in range(iris.data.shape[1] -1):

	clf = neighbors.KNeighborsClassifier(k)
	clf.fit(X, y)
	#xx = np.linspace(4, 8.2)
	
	#xx = np.linspace(X[:, 0].min() - 0.1 * X[:, 0].ptp(), X[:, 0].max() + 0.1 * X[:, 0].ptp(), 1000)
	#yy = np.linspace(X[:, 1].min() - 0.1 * X[:, 1].ptp(), X[:, 1].max() + 0.1 * X[:, 1].ptp(), 1000)
	xx = np.linspace(4, 7.2)
	yy = np.linspace(1, 5)
	plt.figure()

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	
	
	
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	h = 0.002
	xx, yy = np.meshgrid(np.arange(xx.min(), xx.max() + h, h),
	                         np.arange(yy.min(), yy.max() + h, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	
	ax.contour(xx, yy, Z, colors='k', levels=[0, 1], alpha=0.5,
	           linestyles=['-', '-'])
	
	ax.contourf(xx, yy, Z, colors=[MAROON, BLUE, GREEN], levels=[-1, 0, 1, 2], alpha=0.5)           
	print(np.unique(Z.flatten()))  
	
	plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
	plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')
	plt.scatter(X[y == 2, 0], X[y == 2, 1], color=GREEN, label='\emph{Iris virginica}')
	
	plt.legend(loc=0)     
	plt.title('${}-$Nearest Neighbours'.format(k))
	

	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.tight_layout()


	plt.savefig('../figures/knn_{}.pdf'.format(k))
plt.show()
