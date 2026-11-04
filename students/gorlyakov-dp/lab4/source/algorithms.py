import numpy as np


def fit_pca(X):
    mean = np.mean(X, axis=0)
    F = X - mean
    _, singular_values, Ut = np.linalg.svd(F, full_matrices=False)
    return mean, Ut.T, singular_values


def select_pca_dimension(singular_values, epsilon=0.05):
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")

    eigenvalues = singular_values**2
    total = np.sum(eigenvalues)
    if total <= 0:
        raise ValueError("Total variance must be positive")

    errors = np.append(np.cumsum(eigenvalues[::-1])[::-1], 0) / total
    m = int(np.flatnonzero(errors <= epsilon)[0])
    return m, errors
