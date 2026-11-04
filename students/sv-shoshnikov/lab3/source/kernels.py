import numpy as np


def linear_kernel(X, Y):
    return X @ Y.T


def poly_kernel(X, Y, degree=3, coef0=1.0):
    return (X @ Y.T + coef0) ** degree


def rbf_kernel(X, Y, gamma=1.0):
    sq = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1)[None, :] - 2.0 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def make_kernel(name, gamma=1.0, degree=3, coef0=1.0):
    if name == "linear":
        return linear_kernel
    if name == "poly":
        return lambda X, Y: poly_kernel(X, Y, degree, coef0)
    if name == "rbf":
        return lambda X, Y: rbf_kernel(X, Y, gamma)
    raise ValueError(f"Неизвестное ядро: {name}")


def is_positive_semidefinite(K, tol=1e-8):
    return float(np.linalg.eigvalsh((K + K.T) / 2).min()) >= -tol
