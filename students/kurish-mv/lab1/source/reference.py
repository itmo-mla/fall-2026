import numpy as np
from sklearn.linear_model import Ridge

from classifier import LinearClassifier


def analytic_ridge(X, y, l2):
    # (Xa^T Xa + N*l2*R) theta = Xa^T y — минимум того же Q
    n, d = X.shape
    Xa = np.c_[X, np.ones(n)]
    R = np.eye(d + 1)
    R[-1, -1] = 0.0
    theta = np.linalg.solve(Xa.T @ Xa + n * l2 * R, Xa.T @ y)
    return theta[:-1], float(theta[-1])


def sklearn_ridge(X, y, l2):
    m = Ridge(alpha=len(X) * l2).fit(X, y)
    return m.coef_, float(m.intercept_)


def make_model(w, b, l2):
    model = LinearClassifier(l2=l2)
    model.weights, model.bias = w, b
    return model
