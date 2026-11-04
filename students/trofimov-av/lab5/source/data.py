import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(seed=42, features=None):
    iris = load_iris()
    mask = iris.target != 0
    x = iris.data[mask]
    if features is not None:
        x = x[:, features]
    y = iris.target[mask] - 1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (x_train - mean) / scale, (x_test - mean) / scale, y_train, y_test
