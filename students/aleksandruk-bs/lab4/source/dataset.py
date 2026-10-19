"""Калифорнийское жильё: медианная стоимость по восьми числовым признакам.

Датасет берётся из sklearn в момент запуска и в репозиторий не кладётся.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


FEATURE_NAMES = (
    "доход",
    "возраст дома",
    "комнаты",
    "спальни",
    "население",
    "населённость",
    "широта",
    "долгота",
)


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def load_housing(test_size: float = 0.3, seed: int = 42) -> Split:
    """20640 кварталов переписи. Ответ — медианная стоимость в сотнях тысяч долларов."""
    bunch = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(bunch.data, dtype=np.float64),
        np.asarray(bunch.target, dtype=np.float64),
        test_size=test_size,
        random_state=seed,
    )
    return Split(X_train, X_test, y_train, y_test, FEATURE_NAMES)


def correlation(X: np.ndarray) -> np.ndarray:
    """Корреляции столбцов. Перед этим столбцы центрируются."""
    centered = X - X.mean(axis=0)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    standard = centered / scale
    return (standard.T @ standard) / X.shape[0]


def variance_inflation(corr: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R_j^2). R_j^2 берётся из регрессии признака на остальные."""
    precision = np.linalg.inv(corr)
    return np.diag(precision)
