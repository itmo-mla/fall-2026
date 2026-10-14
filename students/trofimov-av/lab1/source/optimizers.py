from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from losses import (
    l2_penalty,
    regularized_squared_loss,
    regularized_squared_loss_gradient,
    sample_squared_loss_gradient,
)
from margins import absolute_margin_order
from quality import update_recursive_quality


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationResult:
    weights: FloatArray
    bias: float
    recursive_quality: FloatArray
    epoch_loss: FloatArray


@dataclass(frozen=True)
class SteepestDescentResult:
    weights: FloatArray
    bias: float
    loss_history: FloatArray
    step_history: FloatArray


def stochastic_gradient_descent_momentum(
    features: ArrayLike,
    labels: ArrayLike,
    initial_weights: ArrayLike | None = None,
    initial_bias: float = 0.0,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    epochs: int = 100,
    smoothing: float = 0.02,
    regularization: float = 0.0,
    shuffle: bool = True,
    presentation: str | None = None,
    random_state: int | None = None,
) -> OptimizationResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Матрица признаков должна быть непустой и двумерной")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("Число меток должно совпадать с числом объектов")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Метки классов должны быть -1 или +1")
    if learning_rate <= 0.0:
        raise ValueError("Скорость обучения должна быть положительной")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("Коэффициент инерции должен принадлежать [0, 1)")
    if epochs <= 0:
        raise ValueError("Количество эпох должно быть положительным")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("Коэффициент сглаживания должен принадлежать (0, 1]")
    if regularization < 0.0:
        raise ValueError("Коэффициент регуляризации не может быть отрицательным")
    if presentation is None:
        presentation = "random" if shuffle else "sequential"
    if presentation not in {"random", "sequential", "margin"}:
        raise ValueError("Неизвестный способ предъявления объектов")

    if initial_weights is None:
        weights = np.zeros(x.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.ndim != 1 or weights.shape[0] != x.shape[1]:
            raise ValueError("Размер начальных весов не совпадает с числом признаков")

    bias = float(initial_bias)
    weights_velocity = np.zeros_like(weights)
    bias_velocity = 0.0
    rng = np.random.default_rng(random_state)
    indices = np.arange(x.shape[0])
    quality_history = np.empty(epochs * x.shape[0], dtype=np.float64)
    epoch_loss_history = np.empty(epochs, dtype=np.float64)
    quality = regularized_squared_loss(x, y, weights, bias, regularization)
    iteration = 0

    for epoch in range(epochs):
        if presentation == "random":
            rng.shuffle(indices)
        elif presentation == "sequential":
            indices = np.arange(x.shape[0])
        else:
            indices = absolute_margin_order(x, y, weights, bias)

        for index in indices:
            weights_gradient, bias_gradient = sample_squared_loss_gradient(
                x, y, weights, bias, int(index)
            )
            current_loss = 0.5 * bias_gradient**2 + l2_penalty(
                weights, regularization
            )
            quality = update_recursive_quality(quality, current_loss, smoothing)
            weights_gradient += regularization * weights

            weights_velocity = (
                momentum * weights_velocity + learning_rate * weights_gradient
            )
            bias_velocity = momentum * bias_velocity + learning_rate * bias_gradient
            weights -= weights_velocity
            bias -= bias_velocity

            quality_history[iteration] = quality
            iteration += 1

        epoch_loss_history[epoch] = regularized_squared_loss(
            x, y, weights, bias, regularization
        )

    return OptimizationResult(
        weights=weights,
        bias=bias,
        recursive_quality=quality_history,
        epoch_loss=epoch_loss_history,
    )


def steepest_gradient_descent(
    features: ArrayLike,
    labels: ArrayLike,
    initial_weights: ArrayLike | None = None,
    initial_bias: float = 0.0,
    regularization: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> SteepestDescentResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Матрица признаков должна быть непустой и двумерной")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("Число меток должно совпадать с числом объектов")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Метки классов должны быть -1 или +1")
    if regularization < 0.0:
        raise ValueError("Коэффициент регуляризации не может быть отрицательным")
    if max_iterations <= 0:
        raise ValueError("Число итераций должно быть положительным")
    if tolerance < 0.0:
        raise ValueError("Допуск не может быть отрицательным")

    if initial_weights is None:
        weights = np.zeros(x.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.ndim != 1 or weights.shape[0] != x.shape[1]:
            raise ValueError("Размер начальных весов не совпадает с числом признаков")

    bias = float(initial_bias)
    losses = [regularized_squared_loss(x, y, weights, bias, regularization)]
    steps: list[float] = []

    for _ in range(max_iterations):
        weights_gradient, bias_gradient = regularized_squared_loss_gradient(
            x, y, weights, bias, regularization
        )
        gradient_norm_squared = float(
            weights_gradient @ weights_gradient + bias_gradient**2
        )
        if gradient_norm_squared <= tolerance**2:
            break

        score_direction = x @ weights_gradient + bias_gradient
        denominator = float(
            np.mean(score_direction**2)
            + regularization * (weights_gradient @ weights_gradient)
        )
        if denominator <= 0.0:
            break

        step = gradient_norm_squared / denominator
        weights -= step * weights_gradient
        bias -= step * bias_gradient
        steps.append(step)
        losses.append(
            regularized_squared_loss(x, y, weights, bias, regularization)
        )

    return SteepestDescentResult(
        weights=weights,
        bias=bias,
        loss_history=np.asarray(losses, dtype=np.float64),
        step_history=np.asarray(steps, dtype=np.float64),
    )
