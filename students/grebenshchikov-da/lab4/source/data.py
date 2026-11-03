"""Загрузка, разбиение и стандартизация данных для эксперимента"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_data() -> PreparedData:
    """Формирует задачу регрессии и стандартизует признаки"""
    dataset = load_wine()
    target_index = dataset.feature_names.index("alcohol")
    X = np.delete(dataset.data, target_index, axis=1)
    y = dataset.data[:, target_index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0

    # Статистики считаются только по train, чтобы избежать утечки
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
