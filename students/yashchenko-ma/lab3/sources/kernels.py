"""Шаг 3: ядра (трюк с ядром)."""
import numpy as np


def linear_kernel(X1, X2):
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma):
    sq1 = np.sum(X1**2, axis=1)[:, None]
    sq2 = np.sum(X2**2, axis=1)[None, :]
    sqdist = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sqdist, 0))


def poly_kernel(X1, X2, degree=3, coef0=1.0):
    return (X1 @ X2.T + coef0) ** degree
