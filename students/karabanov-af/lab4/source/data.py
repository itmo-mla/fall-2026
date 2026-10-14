import numpy as np
from sklearn.datasets import load_diabetes

# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html


def load_data():
    """Diabetes: 442 patients, 10 features, disease progression in a year as the target."""
    data = load_diabetes()
    return data.data, data.target, list(data.feature_names)


def train_test_split(X, y, test_size=0.3, seed=42):
    """Random split: for regression there is nothing to stratify by."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = round(len(y) * test_size)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


def standardize(X, reference=None):
    """Scale by the mean and std of `reference` (the training part) or of X itself."""
    ref = X if reference is None else reference
    return (X - ref.mean(axis=0)) / ref.std(axis=0)
