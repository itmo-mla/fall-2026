import numpy as np
from sklearn.datasets import load_iris


def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def prepare(x_train, x_test, standardize=False):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) if standardize else np.ones(x_train.shape[1])
    scale = np.where(scale == 0, 1.0, scale)
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale
