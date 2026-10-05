import numpy as np

from knn import weighted_vote, gaussian_kernel


def loo_errors(X, y, ks, kernel=gaussian_kernel):
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    classes = np.unique(y)

    errors = []
    for k in ks:
        pred = weighted_vote(dists, y, k, classes, kernel)
        errors.append((pred != y).mean())

    return np.array(errors)
