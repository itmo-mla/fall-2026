import numpy as np
from scipy.special import expit


def predict_probabilities(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.dot(features, weights)
    return expit(scores)


def log_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, scores) - labels * scores))


def residuals_and_variances(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = expit(scores)
    opposite_probabilities = expit(-scores)
    # p - y и p(1 - p), без вычитания близкой к единице вероятности из единицы.
    residuals = np.where(labels == 1, -opposite_probabilities, probabilities)
    variances = probabilities * opposite_probabilities
    return residuals, variances


def loss_gradient(features: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    return np.dot(features.T, residuals) / len(features)


def loss_hessian(features: np.ndarray, variances: np.ndarray) -> np.ndarray:
    return np.dot(features.T, variances[:, None] * features) / len(features)
