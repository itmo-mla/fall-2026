"""Загрузка, разбиение и стандартизация данных"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetSplit:
    """Подготовленные обучающая и тестовая выборки"""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_and_prepare(
    test_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplit:
    """Загружает Raisin и подготавливает данные для эксперимента"""
    dataset = fetch_ucirepo(id=850)
    X = dataset.data.features.to_numpy(dtype=np.float64)
    labels = dataset.data.targets.squeeze().to_numpy(dtype=str)
    y = (labels == "Kecimen").astype(np.int64)

    if X.shape != (900, 7):
        raise ValueError("Ожидалась матрица признаков размером 900 на 7")
    if not np.isfinite(X).all():
        raise ValueError("Данные содержат пропуски или бесконечные значения")
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("Целевая переменная должна содержать классы 0 и 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Статистики считаются только по обучающей выборке, чтобы избежать утечки
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)

    return DatasetSplit(
        X_train=(X_train - mean) / scale,
        X_test=(X_test - mean) / scale,
        y_train=y_train,
        y_test=y_test,
    )
