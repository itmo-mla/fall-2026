import numpy as np
from sklearn.datasets import load_wine

# https://archive.ics.uci.edu/dataset/109/wine

def load_data():
    """Wine: 178 objects, 13 features in different units, 3 classes."""
    wine = load_wine()
    return wine.data, wine.target, list(wine.feature_names), list(wine.target_names)


def train_test_split(X, y, test_size=0.3, seed=42):
    """Stratified split: every class keeps its share in both parts."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = round(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X, reference=None):
    """Scale by the mean and std of `reference` (the training part) or of X itself."""
    ref = X if reference is None else reference
    return (X - ref.mean(axis=0)) / ref.std(axis=0)
