"""Методы второго порядка для логистической регрессии (лекция 5).

Все три метода делают одинаковый шаг w := w − h (Q″)⁻¹ Q′, но вычисляют его по-разному:
- newton_raphson — явно строит гессиан и решает систему Q″ d = Q′ (слайд 7);
- irls          — МНК с весами γ_i = √((1−σ_i)σ_i) и ответами ỹ_i = y_i √((1−σ_i)/σ_i) (слайд 9);
- irls_glm      — IRLS для GLM с распределением Бернулли, y_i ∈ {0, 1} (слайды 18–20).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from losses import gradient, hessian, log_loss, margins, sigmoid

ARMIJO_C = 1e-4


@dataclass
class History:
    w: list[np.ndarray] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    sigma_change: list[float] = field(default_factory=list)  # max |Δσ_i| — критерий остановки слайда 9

    @property
    def n_iter(self) -> int:
        return len(self.step)


def _step_size(F, y, w, direction, grad) -> float:
    """Выбор шага h_t: начинаем с h = 1 (чистый Ньютон) и делим пополам по правилу Армихо.

    Около оптимума всегда принимается h = 1 — тогда сходимость квадратичная.
    """
    h, q = 1.0, log_loss(F, y, w)
    slope = grad @ direction
    while log_loss(F, y, w + h * direction) > q + ARMIJO_C * h * slope and h > 1e-10:
        h /= 2
    return h


def _optimize(F, y, w0, direction_fn: Callable, tol: float, max_iter: int) -> tuple[np.ndarray, History]:
    """Общий цикл: направление спуска от метода, шаг — по Армихо, остановка по ‖Q′(w)‖∞ ≤ tol."""
    w = w0.astype(float).copy()
    hist = History()
    s_prev = sigmoid(margins(F, y, w))

    def record(step: float, s_change: float) -> np.ndarray:
        g = gradient(F, y, w)
        hist.w.append(w.copy())
        hist.loss.append(log_loss(F, y, w))
        hist.grad_norm.append(float(np.max(np.abs(g))))
        if step:
            hist.step.append(step)
            hist.sigma_change.append(s_change)
        return g

    g = record(0.0, 0.0)
    for _ in range(max_iter):
        if hist.grad_norm[-1] <= tol:
            break
        direction = direction_fn(w)
        h = _step_size(F, y, w, direction, g)
        w = w + h * direction
        s = sigmoid(margins(F, y, w))
        g = record(h, float(np.max(np.abs(s - s_prev))))
        s_prev = s
    return w, hist


def newton_raphson(F, y, w0, tol=1e-10, max_iter=100):
    """w := w − h (Q″(w))⁻¹ Q′(w); система решается, гессиан не обращается."""
    def direction(w):
        return -np.linalg.solve(hessian(F, y, w), gradient(F, y, w))
    return _optimize(F, y, w0, direction, tol, max_iter)


def irls(F, y, w0, tol=1e-10, max_iter=100):
    """Слайд 9: w := w + h (F̃ᵀF̃)⁻¹ F̃ᵀ ỹ — решение задачи взвешенного МНК.

    F̃ = diag(γ) F, γ_i = √((1−σ_i)σ_i), ỹ_i = y_i √((1−σ_i)/σ_i) = y_i e^{−M_i/2}.
    Задача ‖F̃ δ − ỹ‖² → min решается через lstsq (SVD), матрица F̃ᵀF̃ не строится.
    """
    def direction(w):
        M = margins(F, y, w)
        s = sigmoid(M)
        gamma = np.sqrt((1.0 - s) * s)
        y_tilde = y * np.exp(-0.5 * M)  # √((1−σ)/σ) = e^{−M/2}: без деления на σ ≈ 0
        delta, *_ = np.linalg.lstsq(gamma[:, None] * F, y_tilde, rcond=None)
        return delta
    return _optimize(F, y, w0, direction, tol, max_iter)


def irls_glm(F, y01, w0, tol=1e-10, max_iter=100):
    """Слайд 19 для распределения Бернулли: θ_i = ⟨w, x_i⟩, c(θ) = ln(1 + e^θ), φ = 1.

    c′(θ) = σ(θ) = μ_i, c″(θ) = μ_i(1 − μ_i); γ_i = √c″(θ_i), ỹ_i = (y_i − μ_i) / γ_i.
    """
    y_pm = 2.0 * y01 - 1.0  # для общего цикла (функционал и градиент записаны через ±1)

    def direction(w):
        mu = sigmoid(F @ w)
        gamma = np.maximum(np.sqrt(mu * (1.0 - mu)), 1e-150)
        delta, *_ = np.linalg.lstsq(gamma[:, None] * F, (y01 - mu) / gamma, rcond=None)
        return delta
    return _optimize(F, y_pm, w0, direction, tol, max_iter)
