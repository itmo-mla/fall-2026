import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_split(test_size=0.2, random_state=42):
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )
    return x_train, x_test, y_train, y_test, iris.feature_names, iris.target_names


def standardize(x_train, x_test):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("Признак с нулевым стандартным отклонением")
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale
