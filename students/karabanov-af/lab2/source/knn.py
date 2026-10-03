import numpy as np


def distances(X_train, X):
    """Euclidean distance from every object of X to every object of X_train: shape (len(X), len(X_train))."""
    return np.linalg.norm(X[:, None, :] - X_train[None, :, :], axis=2)


def gaussian(r):
    """Gaussian kernel K(r) = exp(-r^2 / 2): weight of a neighbour standing at r window widths away."""
    return np.exp(-0.5 * r ** 2)


def predict(D, y_train, k=5):
    """Majority vote of the k nearest neighbours, D is the matrix of distances to the training objects."""
    neighbours = np.argsort(D, axis=1)[:, :k]
    votes = y_train[neighbours]
    return np.array([np.bincount(row, minlength=y_train.max() + 1).argmax() for row in votes])


def class_scores(D, y_train, k=5):
    """Parzen window of variable width: the window h(x) is the distance to the (k + 1)-th neighbour.

    Returns the total weight K(rho / h) of every class for every object: shape (n_classes, len(D)).
    """
    k = min(k, D.shape[1] - 1)
    order = np.argsort(D, axis=1)
    neighbours = order[:, :k]
    rho = np.take_along_axis(D, neighbours, axis=1)
    h = np.take_along_axis(D, order[:, k:k + 1], axis=1)
    weights = gaussian(rho / h)
    return np.array([(weights * (y_train[neighbours] == c)).sum(axis=1) for c in range(y_train.max() + 1)])


def predict_parzen(D, y_train, k=5):
    """The class with the largest total weight of its neighbours."""
    return class_scores(D, y_train, k).argmax(axis=0)


def margins(D, y_train, y, k=5):
    """Margin M(x) = weight of its own class minus the largest weight among the other classes.

    M < 0 means the object is misclassified, and the smaller |M| the closer the object to the boundary.
    """
    scores = class_scores(D, y_train, k)
    own = scores[y, np.arange(len(y))]
    alien = np.where(np.arange(len(scores))[:, None] == y, -np.inf, scores).max(axis=0)
    return own - alien


def loo_risk(X, y, ks, method=predict_parzen):
    """Leave-one-out risk: every object is classified by all the others, the object itself is excluded."""
    D = distances(X, X)
    np.fill_diagonal(D, np.inf)
    return np.array([np.mean(method(D, y, k) != y) for k in ks])
