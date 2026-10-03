import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons

# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

def load_moons(n=300, noise=0.2, seed=42):
    """Two interleaving half circles: 2 features, the classes are not linearly separable."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, 2 * y - 1


def load_cancer():
    """Breast Cancer Wisconsin: 569 objects, 30 features, malignant (-1) or benign (+1)."""
    data = load_breast_cancer()
    return data.data, 2 * data.target - 1


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
