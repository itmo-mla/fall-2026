import numpy as np


def linear_kernel(X, Y):
    return np.asarray(X, dtype=float) @ np.asarray(Y, dtype=float).T


def rbf_kernel(X, Y, gamma=1.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]
    distances = X_norm + Y_norm - 2.0 * X @ Y.T

    return np.exp(-gamma * distances)
