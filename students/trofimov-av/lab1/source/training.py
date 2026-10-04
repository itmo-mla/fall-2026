from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optimizers import OptimizationResult, stochastic_gradient_descent_momentum


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MultistartResult:
    best_result: OptimizationResult
    best_initial_weights: FloatArray
    best_initial_bias: float
    best_start: int
    best_random_state: int
    final_losses: FloatArray


def correlation_initialization(
    features: ArrayLike,
    labels: ArrayLike,
) -> tuple[FloatArray, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Матрица признаков должна быть непустой и двумерной")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("Число меток должно совпадать с числом объектов")

    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean()
    denominator = np.sqrt(
        np.sum(centered_x**2, axis=0) * np.sum(centered_y**2)
    )
    weights = np.divide(
        centered_x.T @ centered_y,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=denominator > 0.0,
    )
    bias = float(y.mean() - weights @ x.mean(axis=0))
    return weights, bias


def random_initialization(
    feature_count: int,
    rng: np.random.Generator,
    scale: float = 0.5,
) -> tuple[FloatArray, float]:
    if feature_count <= 0:
        raise ValueError("Число признаков должно быть положительным")
    if scale <= 0.0:
        raise ValueError("Масштаб инициализации должен быть положительным")
    weights = rng.normal(0.0, scale, size=feature_count)
    bias = float(rng.normal(0.0, scale))
    return weights, bias


def multistart_momentum(
    features: ArrayLike,
    labels: ArrayLike,
    starts: int = 20,
    initialization_scale: float = 0.5,
    random_state: int = 42,
    **optimizer_parameters: object,
) -> MultistartResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Матрица признаков должна быть непустой и двумерной")
    if starts <= 0:
        raise ValueError("Число стартов должно быть положительным")

    rng = np.random.default_rng(random_state)
    final_losses = np.empty(starts, dtype=np.float64)
    best_result: OptimizationResult | None = None
    best_initial_weights: FloatArray | None = None
    best_initial_bias = 0.0
    best_start = -1
    best_seed = -1

    for start in range(starts):
        initial_weights, initial_bias = random_initialization(
            x.shape[1], rng, initialization_scale
        )
        optimizer_seed = random_state + start
        result = stochastic_gradient_descent_momentum(
            x,
            y,
            initial_weights=initial_weights,
            initial_bias=initial_bias,
            random_state=optimizer_seed,
            **optimizer_parameters,
        )
        final_losses[start] = result.epoch_loss[-1]
        if best_result is None or final_losses[start] < best_result.epoch_loss[-1]:
            best_result = result
            best_initial_weights = initial_weights.copy()
            best_initial_bias = initial_bias
            best_start = start
            best_seed = optimizer_seed

    if best_result is None or best_initial_weights is None:
        raise RuntimeError("Не удалось выполнить мультистарт")

    return MultistartResult(
        best_result=best_result,
        best_initial_weights=best_initial_weights,
        best_initial_bias=best_initial_bias,
        best_start=best_start,
        best_random_state=best_seed,
        final_losses=final_losses,
    )
