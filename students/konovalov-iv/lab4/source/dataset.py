import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data(test_size=0.3, seed=42):
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, y_train, y_test, list(data.feature_names)
