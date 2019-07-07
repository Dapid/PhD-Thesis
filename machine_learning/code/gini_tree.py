from sklearn import tree, datasets
import _plot_tree

from settings import MAROON, BLUE, rc

import numpy as np
import pylab as plt

rc('font', **{'size': 20})
iris = datasets.load_iris()
y = iris.target[iris.target < 2]
X = iris.data[iris.target < 2, :2]

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(10, 8))

_plot_tree.plot_tree(clf, class_names=('\emph{Iris setosa}', '\emph{Iris versicolor}'),
                     feature_names=('Sepal \- length', 'Sepal \- width'),
                     filled=True, precision=2, rounded=False, impurity=False, proportion=False)

plt.tight_layout()
plt.savefig('../figures/tree.pdf')

plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')

xx = np.linspace(4, 7.2)
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

ax.contour(xx, yy, Z, colors='k', levels=[0, ], alpha=0.5,
           linestyles=['-'])

ax.contourf(xx, yy, Z, colors=[MAROON, BLUE], levels=[-1, 0, 1], alpha=0.5)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color=MAROON, label='\emph{Iris setosa}')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color=BLUE, label='\emph{Iris versicolor}')

plt.title('Decision tree')

plt.ylim(1, 5)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.tight_layout()

plt.savefig('../figures/tree_2d.pdf')

plt.figure()
plt.subplot(211)

x = X[:, 0]

plt.hist(x[y == 0], bins='auto', color=MAROON, label='\emph{Iris setosa}', histtype='stepfilled', alpha=0.8)
plt.hist(x[y == 1], bins='auto', color=BLUE, label='\emph{Iris versicolor}', histtype='stepfilled', alpha=0.8)
plt.ylabel('Number')

#plt.legend(loc=0)
plt.title('Decision tree: first split')
eps = 0.12
plt.xlim(4-eps, 7+eps)

def gini(vector, features, threshold):
    def _gini(vector):
        p_a = np.mean(vector == 0)
        p_b = np.mean(vector == 1)
        return p_a * (1 - p_a) + p_b * (1 - p_b)

    a = vector[features <= threshold]
    b = vector[features > threshold]
    return (_gini(a) + _gini(b)) / 2


def entropy(vector, features, threshold):
    def _s(vector):
        p_a = np.mean(vector == 0)
        p_b = np.mean(vector == 1)
        eps = 1e-6
        return -p_a * np.log(p_a + eps) - p_b * np.log(p_b + eps)

    a = vector[features <= threshold]
    b = vector[features > threshold]
    return (_s(a) + _s(b)) / 2


x_values = np.linspace(x.min(), x.max(), num=400)
y_values_gini = [gini(y, x, threshold) for threshold in x_values]
y_values_s = [entropy(y, x, threshold) for threshold in x_values]

pos = 5.450
plt.axvline(pos, color='k', ls=':')

plt.subplot(212)
plt.title('Gini impurity')
plt.plot(x_values, y_values_gini, color='k', label='Gini impurity')
# plt.plot(x_values, y_values_s, color='k', ls=':', label='Entropy')
#plt.legend(loc=0)
plt.axvline(pos, color='k', ls=':')
plt.xlabel('Sepal length')
plt.xlim(4-eps, 7+eps)
plt.tight_layout()
plt.savefig('../figures/tree_split.pdf')
plt.show()

plt.show()

estimator = clf
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print(clf.tree_)
