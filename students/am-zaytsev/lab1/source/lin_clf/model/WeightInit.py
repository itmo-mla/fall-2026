import numpy as np

from . import Model


def init_correlation(model: Model, x, y):
    model.w = (x.T.dot(y).T / (x**2).sum(axis=0))[0][:, np.newaxis]
    model.w = np.vstack([model.w, [1]])


def init_random(model: Model, n, m):
    model.w = np.random.uniform(-1 / (2 * n), 1 / (2 * n), m + 1)[:, np.newaxis]
