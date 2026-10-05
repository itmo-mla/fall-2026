import numpy as np


def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2)


def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)


def class_scores(dists, y, k, classes, kernel=gaussian_kernel):
    h = np.sort(dists, axis=1)[:, k - 1]
    h = np.where(h == 0, 1e-9, h)
    weights = kernel(dists / h[:, None])

    scores = np.zeros((dists.shape[0], len(classes)))
    for i, c in enumerate(classes):
        scores[:, i] = weights[:, y == c].sum(axis=1)
    return scores


def weighted_vote(dists, y, k, classes, kernel=gaussian_kernel):
    scores = class_scores(dists, y, k, classes, kernel)
    return classes[np.argmax(scores, axis=1)]


class ParzenKNN:
    def __init__(self, k=5, kernel=gaussian_kernel):
        self.k = k
        self.kernel = kernel

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes = np.unique(self.y_train)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.sqrt(((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        return weighted_vote(dists, self.y_train, self.k, self.classes, self.kernel)
