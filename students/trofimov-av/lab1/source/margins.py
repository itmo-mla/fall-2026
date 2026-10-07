"""Вычисление отступов объектов линейного классификатора."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def decision_function(
    features: ArrayLike, weights: ArrayLike, bias: float
) -> FloatArray:
    """Вычислить значения линейной решающей функции a(x) = <w, x> + b."""

    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("features должен быть двумерной матрицей")
    if w.ndim != 1 or x.shape[1] != w.shape[0]:
        raise ValueError("Размер весов должен совпадать с числом признаков")

    return x @ w + float(bias)


def functional_margin(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> FloatArray:
    """Вычислить функциональные отступы M_i = y_i(<w, x_i> + b)."""

    y = np.asarray(labels, dtype=np.float64)
    scores = decision_function(features, weights, bias)

    if y.ndim != 1 or y.shape[0] != scores.shape[0]:
        raise ValueError("Число меток должно совпадать с числом объектов")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Для вычисления отступа метки должны быть -1 или +1")

    return y * scores


def geometric_margin(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> FloatArray:
    """Вычислить масштабно-инвариантные геометрические отступы."""

    w = np.asarray(weights, dtype=np.float64)
    norm = float(np.linalg.norm(w))
    if norm == 0.0:
        raise ValueError("Геометрический отступ не определён для нулевых весов")
    return functional_margin(features, labels, w, bias) / norm


def centroid_linear_rule(
    features: ArrayLike, labels: ArrayLike
) -> tuple[FloatArray, float]:
    """Построить базовую границу по центрам двух классов.

    Нормаль границы направлена от центра класса -1 к центру класса +1,
    а сама граница проходит через середину между центрами.
    """

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    negative_center = x[y == -1.0].mean(axis=0)
    positive_center = x[y == 1.0].mean(axis=0)
    weights = positive_center - negative_center
    bias = -0.5 * float(weights @ (positive_center + negative_center))
    return weights, bias


def absolute_margin_order(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> NDArray[np.int64]:
    margins = functional_margin(features, labels, weights, bias)
    return np.argsort(np.abs(margins), kind="stable").astype(np.int64)
