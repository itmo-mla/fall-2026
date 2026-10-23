"""Регрессия на главных компонентах (лекция 4, слайд 24) с t×q-fold CV (лекция 6, слайд 21)."""

import numpy as np
from sklearn.model_selection import RepeatedKFold

from pca import PCA


def pcr_fit(X: np.ndarray, y: np.ndarray, m: int) -> tuple[np.ndarray, float]:
    """МНК в новых признаках: β* = D⁻¹ Vᵀ y, α* = U β* = Σ_{j≤m} u_j (v_jᵀ y) / √λ_j.

    Так как G = V D и GᵀG = Λ, β* = Λ⁻¹ Gᵀ y. При m = n это обычный МНК.
    Свободный член — ȳ − x̄ᵀα, потому что PCA работает с центрированной F.
    """
    pca = PCA(m).fit(X)
    G = pca.transform(X)
    y_mean = y.mean()
    beta = (G.T @ (y - y_mean)) / pca.eigenvalues_
    alpha = pca.components_.T @ beta
    return alpha, y_mean - pca.mean_ @ alpha


def pcr_cv_rmse(X: np.ndarray, y: np.ndarray, m: int,
                n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> float:
    """RMSE на t×q-fold CV; PCA обучается только на обучающей части каждого блока."""
    folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    errors = []
    for train, test in folds.split(X):
        alpha, b = pcr_fit(X[train], y[train], m)
        errors.append(np.mean((y[test] - (X[test] @ alpha + b)) ** 2))
    return float(np.sqrt(np.mean(errors)))


def pcr_curve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([pcr_cv_rmse(X, y, m) for m in range(1, X.shape[1] + 1)])
