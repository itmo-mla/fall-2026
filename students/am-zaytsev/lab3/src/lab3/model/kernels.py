import numpy as np


def linear(a, b=None):
    if b is None:
        b = a
    return a @ b.T


def polynomial(a, b=None, d=1):
    if b is None:
        b = a
    return (a @ b.T) ** d


def rbf(a, b=None, gamma=1):
    if b is None:
        b = a
    return np.exp(-gamma * np.linalg.norm(a - b[:, None, :], axis=-1) ** 2)
