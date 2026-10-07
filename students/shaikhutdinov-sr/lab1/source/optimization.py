from dataclasses import dataclass

import numpy as np

from data import Dataset
from model import calculate_margins, l2_gradient, loss_gradient, mean_squared_loss


@dataclass
class FitResult:
    weights: np.ndarray
    epoch_weights: np.ndarray
    quality_history: np.ndarray
    stop_reason: str


def recurrent_quality(
    previous: float,
    loss: float,
    quality_weight: float,
) -> float:
    return (1 - quality_weight) * previous + quality_weight * loss


def sampling_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    strategy: str,
) -> np.ndarray:
    object_count = len(labels)
    if strategy == "uniform":
        return np.full(object_count, 1 / object_count)
    if strategy != "margin":
        raise ValueError("Способ предъявления: uniform или margin")
    absolute_margins = np.abs(calculate_margins(features, labels, weights))

    safe_margins = np.maximum(absolute_margins, np.finfo(float).eps)
    priorities = 1 / safe_margins
    return priorities / priorities.sum()


def steepest_step(
    object_features: np.ndarray,
    gradient: np.ndarray,
    regularization: float,
) -> float:
    score_change = np.dot(object_features, gradient)
    loss_curvature = 2 * score_change ** 2
    feature_gradient = gradient[1:]
    regularization_curvature = regularization * np.dot(feature_gradient, feature_gradient)
    curvature = loss_curvature + regularization_curvature
    if curvature <= 1e-20:
        return 0.0

    return float(np.dot(gradient, gradient) / curvature)


def gradient_step_size(
    squared_norms: np.ndarray,
    regularization: float,
    step_factor: float,
) -> float:
    curvature_bound = 2 * squared_norms.max() + regularization
    return float(step_factor / curvature_bound)


def train_sgd(
    training_data: Dataset,
    initial_weights: np.ndarray,
    *,
    epochs: int = 60,
    regularization: float = 0.01,
    momentum: float = 0.9,
    quality_weight: float = 0.001,
    step_factor: float = 0.7,
    sampling: str = "uniform",
    step_mode: str = "decay",
    seed: int = 2026,
    tolerance: float = 1e-3,
    patience: int = 5,
) -> FitResult:
    if step_mode not in {"decay", "steepest"}:
        raise ValueError("Способ выбора шага: decay или steepest")
    if step_mode == "steepest" and momentum != 0:
        raise ValueError("Точный шаг вдоль градиента требует momentum=0")
    if tolerance < 0 or patience < 1:
        raise ValueError("tolerance должна быть неотрицательной, patience — положительной")

    generator = np.random.default_rng(seed)
    features = training_data.features
    labels = training_data.labels
    object_count = len(labels)
    weights = initial_weights.copy()
    velocity = np.zeros_like(weights)
    quality = mean_squared_loss(features, labels, weights)
    squared_norms = np.sum(features ** 2, axis=1)
    initial_step = gradient_step_size(squared_norms, regularization, step_factor)
    epoch_weights = []
    quality_history = []
    stable_epochs = 0
    stop_reason = "epoch_limit"

    for epoch in range(epochs):
        previous_weights = weights.copy()
        probabilities = sampling_probabilities(
            features,
            labels,
            weights,
            strategy=sampling,
        )
        object_indices = generator.choice(
            object_count,
            size=object_count,
            replace=True,
            p=probabilities,
        )
        decay_step = initial_step / (1 + epoch) ** 0.6

        for index in object_indices:
            object_features = features[index]
            label = labels[index]
            error = float(np.dot(object_features, weights) - label)
            loss = error ** 2

            gradient = loss_gradient(object_features, error)
            gradient += l2_gradient(weights, regularization)

            if step_mode == "steepest":
                step = steepest_step(
                    object_features,
                    gradient,
                    regularization,
                )
            else:
                step = decay_step

            velocity *= momentum
            velocity += (1 - momentum) * gradient
            weights -= step * velocity

            quality = recurrent_quality(quality, loss, quality_weight)

        epoch_weights.append(weights.copy())
        quality_history.append(quality)

        weight_change = np.linalg.norm(weights - previous_weights)
        weight_scale = max(1.0, np.linalg.norm(previous_weights))

        if weight_change <= tolerance * weight_scale:
            stable_epochs += 1
        else:
            stable_epochs = 0
        if stable_epochs >= patience:
            stop_reason = "weights_stabilized"
            break

    assert np.isfinite(weights).all()
    return FitResult(
        weights=weights,
        epoch_weights=np.asarray(epoch_weights),
        quality_history=np.asarray(quality_history),
        stop_reason=stop_reason,
    )
