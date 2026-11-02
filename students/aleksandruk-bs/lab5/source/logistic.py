"""Логистическая регрессия: Ньютон–Рафсон и IRLS.

Метки y_i ∈ {−1, +1}. Функционал

    Q(w) = Σ ln(1 + exp(−y_i w·x_i)).

σ_i = 1 / (1 + exp(−y_i w·x_i)).

    ∂Q/∂w = −Fᵀ (y ⊙ (1 − σ)),
    ∂²Q     = Fᵀ diag(σ ⊙ (1 − σ)) F.

Шаг Ньютона при h = 1:

    w := w − H⁻¹ ∇Q = w + (Fᵀ W F)⁻¹ Fᵀ (y ⊙ (1 − σ)),
    W = diag(σ ⊙ (1 − σ)).

IRLS — та же линейная система, записанная как взвешенный МНК.
γ_i = √(σ_i (1 − σ_i)), F~ = diag(γ) F,
модифицированный ответ z_i = y_i √((1 − σ_i) / σ_i) = y_i (1 − σ_i) / γ_i.

Тогда F~ᵀ F~ = H и F~ᵀ z = Fᵀ (y ⊙ (1 − σ)), поэтому направление совпадает
с ньютоновским. Деление на √σ без множителя (1 − σ) дало бы Fᵀ y и другой шаг.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class FitResult:
    w: np.ndarray
    loss: np.ndarray
    path: np.ndarray
    steps: np.ndarray
    n_iter: int


def log_loss(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    margins = y * (F @ w)
    return float(np.sum(np.logaddexp(0.0, -margins)))


def class_probability(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P(y = +1 | x) = σ(w·x). Свободный член уже стоит в F."""
    return expit(F @ w)


def predict_signed(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.where(F @ w >= 0.0, 1.0, -1.0)


def _sigma(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """σ(y w·x), обрезанная от 0 и 1, чтобы вес и модифицированный ответ были конечными."""
    sigma = expit(y * (F @ w))
    return np.clip(sigma, 1e-12, 1.0 - 1e-12)


def newton_direction(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """−H⁻¹ ∇Q."""
    sigma = _sigma(F, y, w)
    residual = y * (1.0 - sigma)
    hess_weight = sigma * (1.0 - sigma)
    gradient = -F.T @ residual
    hessian = F.T @ (hess_weight[:, None] * F)
    return np.linalg.solve(hessian, -gradient)


def irls_direction(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(F~ᵀ F~)⁻¹ F~ᵀ z, см. модульный комментарий."""
    sigma = _sigma(F, y, w)
    gamma = np.sqrt(sigma * (1.0 - sigma))
    modified = y * np.sqrt((1.0 - sigma) / sigma)
    weighted = gamma[:, None] * F
    gram = weighted.T @ weighted
    rhs = weighted.T @ modified
    return np.linalg.solve(gram, rhs)


def fit_logistic(
    F: np.ndarray,
    y: np.ndarray,
    method: str = "newton",
    step: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 30,
) -> FitResult:
    """w⁰ = 0. Останов, когда ||w^{t+1} − w^t|| < tol."""
    if method == "newton":
        direction = newton_direction
    elif method == "irls":
        direction = irls_direction
    else:
        raise ValueError(f"неизвестный метод: {method}")

    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(F.shape[1], dtype=np.float64)
    path = [w.copy()]
    losses = [log_loss(F, y, w)]
    steps: list[float] = []
    for _ in range(max_iter):
        delta = direction(F, y, w)
        move = step * delta
        length = float(np.linalg.norm(move))
        steps.append(length)
        if length < tol:
            break
        w = w + move
        path.append(w.copy())
        losses.append(log_loss(F, y, w))
    return FitResult(
        w=w,
        loss=np.asarray(losses, dtype=np.float64),
        path=np.vstack(path),
        steps=np.asarray(steps, dtype=np.float64),
        n_iter=len(steps),
    )
