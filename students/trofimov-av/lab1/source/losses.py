"""Квадратичная функция потерь и её градиент."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from margins import decision_function


FloatArray = NDArray[np.float64]


def _validated_labels(labels: ArrayLike, object_count: int) -> FloatArray:
    """Проверить и вернуть бинарные метки в виде массива float64."""

    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] != object_count:
        raise ValueError("Число меток должно совпадать с числом объектов")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Метки классов должны быть -1 или +1")
    return y


def squared_loss(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> float:
    """Вычислить среднюю квадратичную потерю.

    Функция записывается через отступ как
    Q = 1/(2n) * sum((1 - M_i) ** 2), где M_i = y_i * a(x_i).
    Для меток {-1, +1} она эквивалентна
    Q = 1/(2n) * sum((a(x_i) - y_i) ** 2).
    Множитель 1/2 упрощает выражение градиента.
    """

    scores = decision_function(features, weights, bias)
    y = _validated_labels(labels, scores.shape[0])
    errors = scores - y
    return 0.5 * float(np.mean(errors**2))


def squared_loss_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> tuple[FloatArray, float]:
    """Вычислить градиент квадратичной потери по w и b.

    dQ/dw = X.T @ (Xw + b - y) / n
    dQ/db = mean(Xw + b - y)
    """

    x = np.asarray(features, dtype=np.float64)
    scores = decision_function(x, weights, bias)
    y = _validated_labels(labels, scores.shape[0])
    errors = scores - y

    weights_gradient = x.T @ errors / x.shape[0]
    bias_gradient = float(np.mean(errors))
    return weights_gradient, bias_gradient


def sample_squared_loss_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
    index: int,
) -> tuple[FloatArray, float]:
    """Вычислить градиент потери одного объекта для будущего SGD."""

    x = np.asarray(features, dtype=np.float64)
    y = _validated_labels(labels, x.shape[0])
    if not 0 <= index < x.shape[0]:
        raise IndexError("Индекс объекта находится вне выборки")

    score = float(decision_function(x[index : index + 1], weights, bias)[0])
    error = score - y[index]
    return error * x[index], float(error)


def l2_penalty(weights: ArrayLike, regularization: float) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if regularization < 0.0:
        raise ValueError("Коэффициент регуляризации не может быть отрицательным")
    return 0.5 * regularization * float(w @ w)


def regularized_squared_loss(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
    regularization: float,
) -> float:
    return squared_loss(features, labels, weights, bias) + l2_penalty(
        weights, regularization
    )


def regularized_squared_loss_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    bias: float,
    regularization: float,
) -> tuple[FloatArray, float]:
    w = np.asarray(weights, dtype=np.float64)
    weights_gradient, bias_gradient = squared_loss_gradient(
        features, labels, w, bias
    )
    if regularization < 0.0:
        raise ValueError("Коэффициент регуляризации не может быть отрицательным")
    return weights_gradient + regularization * w, bias_gradient
