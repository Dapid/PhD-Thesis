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


iris = datasets.load_iris()
y = iris.target[iris.target < 2]
X = iris.data[iris.target < 2, :2]


for k in (1, 3, 5):

	clf = neighbors.KNeighborsClassifier(k)
	clf.fit(X, y)
	xx = np.linspace(4, 7.2)
	
	plt.figure()
	plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
	plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')
	
	plt.xlim(xx.min(), xx.max())
	plt.ylim(1, 5)
	
	plt.legend(loc=0)
	
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	h = 0.002
	xx, yy = np.meshgrid(np.arange(xx.min(), xx.max() + h, h),
	                         np.arange(1, 5 + h, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	
	ax.contour(xx, yy, Z, colors='k', levels=[0,], alpha=0.5,
	           linestyles=['-'])
	
	ax.contourf(xx, yy, Z, colors=[MAROON, BLUE], levels=[-1, 0, 1], alpha=0.5)           
	print(Z)
	print(np.unique(Z.flatten()))  
	
	plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
	plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')
	         
	plt.title('${}-$Nearest Neighbours'.format(k))
	
	plt.ylim(1, 5)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.tight_layout()

	plt.savefig('../figures/knn_{}.pdf'.format(k))
plt.show()
