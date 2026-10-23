import numpy as np


def gaussian_kernel(r):
    return np.exp(-2 * r ** 2)


def pairwise_distances(A, B):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0))


def parzen_knn_predict(dist, y_ref, k, n_classes, kernel=gaussian_kernel):
    order = np.argsort(dist, axis=1, kind="stable")[:, :k + 1]
    d = np.take_along_axis(dist, order, axis=1)
    h = np.maximum(d[:, k], 1e-12)
    weights = kernel(d[:, :k] / h[:, None])
    neighbor_labels = y_ref[order[:, :k]]
    votes = np.stack([(weights * (neighbor_labels == c)).sum(axis=1) for c in range(n_classes)], axis=1)
    return votes.argmax(axis=1)


def loo_risks(X, y, ks, ref_mask=None, biased=False):
    n_classes = len(np.unique(y))
    D = pairwise_distances(X, X)
    if ref_mask is not None:
        D[:, ~ref_mask] = np.inf
    if not biased:
        np.fill_diagonal(D, np.inf)
    return np.array([np.mean(parzen_knn_predict(D, y, k, n_classes) != y) for k in ks])


class ParzenKNN:
    def __init__(self, k=5, kernel=gaussian_kernel):
        self.k = k
        self.kernel = kernel

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_classes = int(y.max()) + 1
        return self

    def predict(self, X):
        dist = pairwise_distances(X, self.X)
        return parzen_knn_predict(dist, self.y, self.k, self.n_classes, self.kernel)


def one_nn_predict(X, X_ref, y_ref):
    return y_ref[np.argmin(pairwise_distances(X, X_ref), axis=1)]
