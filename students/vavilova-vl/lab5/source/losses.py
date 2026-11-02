"""Функционал логистической регрессии, его градиент и гессиан (лекция 5, слайды 6–7).

Метки y_i ∈ {−1, +1}, отступ M_i = y_i ⟨w, x_i⟩, σ_i = σ(M_i).
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^{−z}); запись через tanh не переполняется при больших |z|."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def margins(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return y * (F @ w)


def log_loss(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Q(w) = Σ ln(1 + e^{−M_i}); logaddexp(0, −M) = ln(e⁰ + e^{−M}) считается без переполнения."""
    return float(np.sum(np.logaddexp(0.0, -margins(F, y, w))))


def gradient(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """∂Q/∂w_j = −Σ (1 − σ_i) y_i f_j(x_i)."""
    s = sigmoid(margins(F, y, w))
    return -F.T @ ((1.0 - s) * y)


def hessian(F: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """∂²Q/∂w_j∂w_k = Σ (1 − σ_i) σ_i f_j(x_i) f_k(x_i), то есть Fᵀ D F, D = diag((1 − σ_i) σ_i)."""
    s = sigmoid(margins(F, y, w))
    return F.T @ (F * ((1.0 - s) * s)[:, None])
