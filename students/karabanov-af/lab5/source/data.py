import numpy as np
from sklearn.datasets import load_breast_cancer

# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


def load_data():
    """Breast cancer wisconsin: 569 tumours, 30 features, malignant or benign."""
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names)


def train_test_split(X, y, test_size=0.3, seed=42):
    """Stratified split: the class ratio of the whole sample is kept in both parts."""
    rng = np.random.default_rng(seed)
    test = np.concatenate([rng.permutation(np.flatnonzero(y == label))[:round((y == label).sum() * test_size)]
                           for label in np.unique(y)])
    train = np.setdiff1d(np.arange(len(y)), test)
    return X[train], X[test], y[train], y[test]


def standardize(X, reference=None):
    """Scale by the mean and std of `reference` (the training part) or of X itself."""
    ref = X if reference is None else reference
    return (X - ref.mean(axis=0)) / ref.std(axis=0)


def add_bias(X):
    """Append a constant feature so that the free term is just one more weight."""
    return np.column_stack([np.ones(len(X)), X])
