import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def update_recursive_quality(
    previous_quality: float,
    current_loss: float,
    smoothing: float,
) -> float:
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("Коэффициент сглаживания должен принадлежать (0, 1]")
    if not np.isfinite(previous_quality) or not np.isfinite(current_loss):
        raise ValueError("Значения качества и потери должны быть конечными")
    return (1.0 - smoothing) * previous_quality + smoothing * current_loss


def recursive_quality_history(
    losses: ArrayLike,
    smoothing: float,
    initial_quality: float | None = None,
) -> FloatArray:
    loss_values = np.asarray(losses, dtype=np.float64)
    if loss_values.ndim != 1 or loss_values.size == 0:
        raise ValueError("Последовательность потерь должна быть непустым вектором")
    if not np.all(np.isfinite(loss_values)):
        raise ValueError("Все значения потерь должны быть конечными")

    quality = float(loss_values[0] if initial_quality is None else initial_quality)
    history = np.empty_like(loss_values)
    for index, current_loss in enumerate(loss_values):
        quality = update_recursive_quality(quality, float(current_loss), smoothing)
        history[index] = quality
    return history
