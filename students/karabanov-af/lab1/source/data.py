import numpy as np
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
POSITIVE_CLASS = "virginica"
NEGATIVE_CLASS = "versicolor"


def load_binary_iris():
    """Versicolor vs virginica: the only pair of iris classes that is not linearly separable."""
    iris = load_iris()
    species = iris.target_names[iris.target]
    keep = np.isin(species, [POSITIVE_CLASS, NEGATIVE_CLASS])
    X = iris.data[keep]
    y = np.where(species[keep] == POSITIVE_CLASS, 1, -1)
    return X, y


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


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])
