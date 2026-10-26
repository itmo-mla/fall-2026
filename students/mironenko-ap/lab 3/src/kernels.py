import numpy as np


def linear_kernel(X, Y):
    return X @ Y.T


def rbf_kernel(X, Y, gamma=1.0):
    X2 = np.sum(X ** 2, axis=1)[:, None]
    Y2 = np.sum(Y ** 2, axis=1)[None, :]
    D2 = X2 + Y2 - 2.0 * X @ Y.T
    return np.exp(-gamma * np.maximum(D2, 0.0))


def poly_kernel(X, Y, degree=3, coef0=1.0):
    return (X @ Y.T + coef0) ** degree


KERNELS = {
    "linear": linear_kernel,
    "rbf": rbf_kernel,
    "poly": poly_kernel,
}


def get_kernel(name):
    if name not in KERNELS:
        raise ValueError(f"Неизвестное ядро: {name}. Доступно: {list(KERNELS)}")
    return KERNELS[name]