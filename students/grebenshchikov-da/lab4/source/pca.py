"""Ручная реализация PCA через сингулярное разложение матрицы"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Переводит объекты в координаты главных компонент"""
        count = len(self.components) if n_components is None else n_components
        return (X - self.mean) @ self.components[:count].T

    def inverse_transform(self, projection: np.ndarray) -> np.ndarray:
        """Восстанавливает признаки по сохранённым компонентам"""
        count = projection.shape[1]
        return projection @ self.components[:count] + self.mean


def fit_pca(X: np.ndarray) -> PCA:
    """Строит PCA через SVD центрированной матрицы признаков"""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    # Центрируем признаки перед SVD
    centered = X - mean

    # Строки Vt задают направления главных компонент
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    # Квадраты сингулярных чисел задают дисперсию компонент
    explained_variance = singular_values**2 / (len(X) - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return PCA(
        mean=mean,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
    )


def effective_dimension(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    """Возвращает минимальное число компонент с заданной долей дисперсии"""
    cumulative = np.cumsum(explained_variance_ratio)
    return int(np.searchsorted(cumulative, threshold) + 1)


def relative_reconstruction_error(model: PCA, X: np.ndarray, count: int) -> float:
    """Считает долю квадрата нормы, потерянную после восстановления"""
    centered = X - model.mean
    projection = model.transform(X, count)
    restored = model.inverse_transform(projection) - model.mean
    return float(np.sum((centered - restored) ** 2) / np.sum(centered**2))
