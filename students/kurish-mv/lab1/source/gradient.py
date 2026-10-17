import numpy as np

from loss import objective
from margin import calculate_margin


def calculate_gradient(X, y, w, b=0.0, l2=0.0):
    # градиент Q: среднее по объектам + L2; для одного объекта передаём X[i:i+1]
    c = -2.0 * (1.0 - calculate_margin(X, y, w, b)) * y
    return X.T @ c / len(X) + 2.0 * l2 * w, float(c.mean())


def numerical_gradient(X, y, w, b=0.0, l2=0.0, eps=1e-5):
    theta = np.append(w, b)
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        d = np.zeros_like(theta)
        d[j] = eps
        hi, lo = theta + d, theta - d
        grad[j] = (objective(X, y, hi[:-1], hi[-1], l2)
                   - objective(X, y, lo[:-1], lo[-1], l2)) / (2 * eps)
    return grad[:-1], float(grad[-1])
