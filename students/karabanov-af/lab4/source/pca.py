import numpy as np


def fit(X):
    """PCA through the SVD of the centered matrix: X - mean = U @ diag(s) @ Vt.

    The rows of Vt are the principal axes, ordered by the singular values s.
    """
    mean = X.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt, singular_values


def explained_variance(singular_values, n_objects):
    """Variance along every principal axis: s_j^2 / (n - 1)."""
    return singular_values ** 2 / (n_objects - 1)


def explained_variance_ratio(singular_values):
    """Share of the total variance that every principal axis carries."""
    return singular_values ** 2 / np.sum(singular_values ** 2)


def transform(X, mean, components, k=None):
    """Coordinates of the objects in the basis of the first k principal axes."""
    return (X - mean) @ components[:k].T


def inverse_transform(P, mean, components):
    """Back to the feature space from the k coordinates kept in P."""
    return P @ components[:P.shape[1]] + mean


def residual_share(ratio):
    """Share of the squared norm that is lost when only the first m axes are kept.

    E_m = (lambda_{m+1} + ... + lambda_n) / (lambda_1 + ... + lambda_n).
    """
    return 1 - np.cumsum(ratio)


def effective_dimension(ratio, eps=0.05):
    """Effective dimension of the sample: the smallest m with E_m <= eps."""
    return int(np.searchsorted(-residual_share(ratio), -eps) + 1)


def steep_slope(residual):
    """Ratios E_(m-1) / E_m for m = 1 .. d - 1, where the criterion looks for a jump."""
    previous = np.concatenate(([1.0], residual[:-1]))
    return previous[:-1] / residual[:-1]
