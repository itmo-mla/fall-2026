"""Загрузка и предобработка Wine (UCI, через sklearn.datasets)."""
import numpy as np
from pathlib import Path

# sklearn используется ТОЛЬКО для загрузки данных.
# Реализация KNN, LOO и отбора эталонов — на numpy.
from sklearn.datasets import load_wine as _sk_load_wine


def load_raw():
    """
    Возвращает объект с полями:
        X : (n, d) float64
        y : (n,)   int
        target_names : список строковых меток классов
        feature_names : список имён признаков
    """
    data = _sk_load_wine()
    return data


def make_xy(data, grouping=None):
    """
        Извлекает X, y, classes из объекта sklearn.
        grouping не используется (3 класса уже сбалансированы),
        но параметр оставлен для совместимости сигнатуры.
    """
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    classes = list(data.target_names)
    return X, y, classes


def train_test_split_np(X, y, test_size=0.2, random_state=42):
    """Стратифицированный split на чистом numpy."""
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_test = max(1, int(round(len(idx_c) * test_size)))
        test_idx.extend(idx_c[:n_test].tolist())
        train_idx.extend(idx_c[n_test:].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma