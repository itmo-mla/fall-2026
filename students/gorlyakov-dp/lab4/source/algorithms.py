import numpy as np


def fit_pca(X):
    mean = np.mean(X, axis=0)
    F = X - mean
    _, singular_values, Ut = np.linalg.svd(F, full_matrices=False)
    return mean, Ut.T, singular_values
