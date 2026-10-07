"""PCA и регрессия через сингулярное разложение, в обозначениях лекции.

Центрированная (и при необходимости нормированная) матрица:

    F = V D U^T

U (n × n) — правые сингулярные векторы, собственные векторы F^T F.
V (ℓ × n) — левые сингулярные векторы.
D = diag(√λ_1, ..., √λ_n), λ_1 ≥ ... ≥ λ_n ≥ 0.

Новые признаки G = F U = V D. Восстановление F_hat = G U^T.
Ошибка по Фробениусу:

    E_m = ||G_m U_m^T - F||^2 / ||F||^2 = (сумма λ_j от j=m+1 до n) / (сумма всех λ).

МНК в главных компонентах, ответ y уже центрирован:

    β* = D^{-1} V^T y,    ŷ = G β*.

Гребневая регрессия на том же разложении:

    β_j = σ_j / (σ_j^2 + α) * (V^T y)_j,    β = U β_компонент.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAModel:
    mean: np.ndarray
    scale: np.ndarray
    U: np.ndarray
    V: np.ndarray
    sigma: np.ndarray
    lam: np.ndarray
    G: np.ndarray
    F: np.ndarray

    def project(self, X: np.ndarray, m: int | None = None) -> np.ndarray:
        take = self.U.shape[1] if m is None else m
        return self.transform(X) @ self.U[:, :take]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.scale

    def reconstruct(self, m: int) -> np.ndarray:
        return self.G[:, :m] @ self.U[:, :m].T


def fit_pca(X: np.ndarray, scale: bool = True) -> PCAModel:
    """SVD матрицы после вычитания среднего. scale=True делит столбцы на их СКО."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    column_scale = np.sqrt(np.mean(centered * centered, axis=0))
    if not scale:
        column_scale = np.ones_like(column_scale)
    column_scale = np.where(column_scale > 0, column_scale, 1.0)
    F = centered / column_scale
    # numpy: F = U_np @ diag(σ) @ Vt. В лекции V — левые векторы, U — правые.
    V, sigma, Ut = np.linalg.svd(F, full_matrices=False)
    U = Ut.T
    lam = sigma ** 2
    G = F @ U
    return PCAModel(mean, column_scale, U, V, sigma, lam, G, F)


def approximation_error(lam: np.ndarray) -> np.ndarray:
    """E_m для m = 0, 1, ..., n. E_0 = 1, E_n = 0."""
    total = float(lam.sum())
    kept = np.cumsum(lam)
    tail = (total - kept) / total
    return np.concatenate([np.array([1.0]), tail])


def effective_dimension(errors: np.ndarray, eps: float = 0.05) -> int:
    """Наименьшее m, при котором E_m ≤ ε."""
    hits = np.flatnonzero(errors <= eps)
    if len(hits) == 0:
        return int(len(errors) - 1)
    return int(hits[0])


def scree_dimension(lam: np.ndarray) -> int:
    """m после самого резкого перепада спектра: максимум λ_m / λ_{m+1}."""
    ratios = lam[:-1] / np.maximum(lam[1:], 1e-15)
    # индекс 0 означает разрыв между первой и второй компонентой, то есть m = 1
    return int(np.argmax(ratios) + 1)


def ols_coefficients(model: PCAModel, y: np.ndarray, m: int) -> np.ndarray:
    """β* = D_m^{-1} V_m^T (y - ȳ). Длина вектора равна m."""
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    return (model.V[:, :m].T @ y_centered) / model.sigma[:m]


def predict_ols(model: PCAModel, beta: np.ndarray, X: np.ndarray, y_mean: float) -> np.ndarray:
    G = model.project(X, m=len(beta))
    return G @ beta + y_mean


def ridge_coefficients(model: PCAModel, y: np.ndarray, alpha: float) -> np.ndarray:
    """Коэффициенты в исходных нормированных признаках: β = U γ, γ_j = σ_j / (σ_j^2 + α) (V^T y)_j."""
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    gamma = model.sigma / (model.lam + alpha) * (model.V.T @ y_centered)
    return model.U @ gamma


def predict_ridge(model: PCAModel, beta: np.ndarray, X: np.ndarray, y_mean: float) -> np.ndarray:
    return model.transform(X) @ beta + y_mean


def gcv_scores(model: PCAModel, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """GCV(α) = ||y - ŷ||^2 / (ℓ - tr H)^2. tr H = сумма σ_j^2 / (σ_j^2 + α)."""
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    rhs = model.V.T @ y_centered
    ell = float(len(y_centered))
    scores = np.empty(len(alphas), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        shrink = model.lam / (model.lam + alpha)
        residual = y_centered - model.V @ (shrink * rhs)
        # +1: свободный член, среднее ответа, оценивается отдельно от гребня.
        denom = ell - 1.0 - float(shrink.sum())
        scores[i] = float(residual @ residual) / denom ** 2
    return scores
