import numpy as np


def decision_function(
    features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.dot(features, weights)


def predict_labels(scores: np.ndarray) -> np.ndarray:
    return np.where(scores >= 0, 1, -1)


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = decision_function(features, weights)
    return predict_labels(scores)


def calculate_margins(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scores = decision_function(features, weights)
    return labels * scores


def squared_losses(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scores = decision_function(features, weights)
    errors = scores - labels
    return errors ** 2


def mean_squared_loss(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    losses = squared_losses(features, labels, weights)
    return float(losses.mean())


def loss_gradient(
    object_features: np.ndarray,
    error: float,
) -> np.ndarray:
    return 2 * error * object_features


def l2_gradient(weights: np.ndarray, regularization: float) -> np.ndarray:
    gradient = regularization * weights

    gradient[0] = 0
    return gradient


def l2_penalty(weights: np.ndarray, regularization: float) -> float:
    feature_weights = weights[1:]
    squared_norm = np.dot(feature_weights, feature_weights)
    return float(regularization / 2 * squared_norm)


def objective(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> float:
    mean_loss = mean_squared_loss(features, labels, weights)
    penalty = l2_penalty(weights, regularization)
    return mean_loss + penalty
