import numpy as np

import matplotlib.pyplot as plt
import GPy

from settings import MAROON, BLUE

np.random.seed(42)

x_true = np.linspace(-4, 4, num=1000)
y_true = np.sin(x_true)

N = 50
x = np.random.random(N) * (2 * np.pi) - np.pi
X = x[:, None]
y = np.sin(X) + np.random.randn(N)[:, None] * 0.1

m = GPy.models.GPRegression(X, y)

m.optimize()

plt.scatter(X, y, color='k', marker='s', s=10)
pred, var_ = m.predict(x_true[:, None])

pred = pred.squeeze()
std = np.sqrt(var_.squeeze())

plt.plot(x_true, y_true, color='k', alpha=0.8, label='True')
plt.plot(x_true, pred.squeeze(), color=MAROON, label='Prediction')
plt.fill_between(x_true, pred - std, pred + std, alpha=0.4, color=MAROON, label='$\pm 1\sigma$')

plt.legend(loc=4)
plt.title('Gaussian Processes')
plt.xlim(-4, 4)
plt.ylim(-1.4, 1.4)
plt.xlabel('x')
plt.ylabel(r'$\sin(x)$')
plt.tight_layout()
plt.savefig('../figures/sin_toy.pdf')

plt.figure()
plt.plot(x_true, y_true, color='k', alpha=0.8, label='True')
plt.plot(x_true, m.posterior_samples_f(x_true[:, None], 4).squeeze(), color=MAROON, label='Samples', alpha=0.8, ls='-.')
plt.title('Gaussian Process sampling')
plt.xlim(-4, 4)
plt.ylim(-1.4, 1.4)
plt.xlabel('x')
plt.ylabel(r'$\sin(x)$')
plt.tight_layout()
plt.savefig('../figures/sin_samples.pdf')

plt.figure()
valid = np.squeeze((X < 0.4) | (X > 2.5))
X = X[valid, :]
y = y[valid, :]

m = GPy.models.GPRegression(X, y)
m.optimize()

plt.scatter(X, y, color='k', marker='s', s=10)
pred, var_ = m.predict(x_true[:, None])

pred = pred.squeeze()
std = np.sqrt(var_.squeeze())

plt.plot(x_true, y_true, color='k', alpha=0.8, label='True')
plt.plot(x_true, pred.squeeze(), color=MAROON, label='Prediction')
plt.fill_between(x_true, pred - std, pred + std, alpha=0.4, color=MAROON, label='$\pm 1\sigma$')

plt.legend(loc=4)
plt.title('Gaussian Processes, gapped')
plt.xlim(-4, 4)
plt.ylim(-1.4, 1.4)
plt.xlabel('x')
plt.ylabel(r'$\sin(x)$')
plt.tight_layout()
plt.savefig('../figures/sin_toy_gapped.pdf')

plt.show()
