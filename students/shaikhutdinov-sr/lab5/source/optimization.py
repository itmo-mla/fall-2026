from dataclasses import dataclass

import numpy as np

from model import (
    loss_hessian,
    log_loss,
    loss_gradient,
    residuals_and_variances,
)


@dataclass
class FitResult:
    weights: np.ndarray
    weight_history: np.ndarray
    loss_history: np.ndarray
    gradient_history: np.ndarray
    converged: bool

    @property
    def iterations(self) -> int:
        return len(self.loss_history) - 1


def newton_step(
    features: np.ndarray,
    weights: np.ndarray,
    gradient: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    hessian = loss_hessian(features, variances)
    increment = np.linalg.solve(hessian, -gradient)
    return weights + increment


def irls_step(
    features: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    # Рабочий ответ z = scores - residuals / variances.
    # Сразу вычисляем Wz: это та же система, без деления на нулевые веса.
    weighted_response = variances * scores - residuals
    normal_matrix = np.dot(features.T, variances[:, None] * features)
    right_hand_side = np.dot(features.T, weighted_response)
    return np.linalg.solve(normal_matrix, right_hand_side)


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
    *,
    tolerance: float = 1e-10,
    max_iter: int = 100,
) -> FitResult:
    if method not in {"newton", "irls"}:
        raise ValueError("Метод должен быть newton или irls")
    if tolerance <= 0 or max_iter < 1:
        raise ValueError("Допуск и лимит итераций должны быть положительными")

    weights = np.zeros(features.shape[1])
    weight_history = []
    loss_history = []
    gradient_history = []
    converged = False

    for iteration in range(max_iter + 1):
        scores = np.dot(features, weights)
        residuals, variances = residuals_and_variances(scores, labels)
        gradient = loss_gradient(features, residuals)
        gradient_norm = np.linalg.norm(gradient, ord=np.inf)
        weight_history.append(weights.copy())
        loss_history.append(log_loss(scores, labels))
        gradient_history.append(gradient_norm)

        if gradient_norm <= tolerance:
            converged = True
            break
        if iteration == max_iter:
            break

        if method == "newton":
            weights = newton_step(features, weights, gradient, variances)
        else:
            weights = irls_step(features, scores, residuals, variances)
        if not np.isfinite(weights).all():
            raise FloatingPointError("Получены нечисловые коэффициенты модели")

    return FitResult(
        weights=weights,
        weight_history=np.asarray(weight_history),
        loss_history=np.asarray(loss_history),
        gradient_history=np.asarray(gradient_history),
        converged=converged,
    )
