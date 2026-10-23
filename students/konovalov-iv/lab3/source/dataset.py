import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def to_pm1(y):
    return np.where(y == 1, 1.0, -1.0)


def _split(X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), to_pm1(y_train), to_pm1(y_test)


def load_moons(n=200, noise=0.25, test_size=0.3, seed=42):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return _split(X, y, test_size, seed)


def load_cancer(test_size=0.3, seed=42):
    data = load_breast_cancer()
    return _split(data.data, data.target, test_size, seed)
