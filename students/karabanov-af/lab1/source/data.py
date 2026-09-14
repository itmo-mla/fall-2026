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


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])
