import numpy as np


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Линейное ядро: K(A,B) = A @ Bᵀ"""
    return A @ B.T


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """RBF-ядро (гауссово): K(A,B) = exp(−γ‖a−b‖²)"""
    aa = np.sum(A ** 2, axis=1)[:, np.newaxis]
    bb = np.sum(B ** 2, axis=1)[np.newaxis, :]
    sq_dist = np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)
    return np.exp(-gamma * sq_dist)


def compute_kernel(X: np.ndarray, Y: np.ndarray, kind: str,
                   gamma: float = 1.0) -> np.ndarray:
    """Выбор ядра по имени. kind='linear' или kind='rbf'"""
    if kind == "linear":
        return linear_kernel(X, Y)
    if kind == "rbf":
        return rbf_kernel(X, Y, gamma)
    raise ValueError(f"Неизвестное ядро: {kind!r}; поддерживаются 'linear' и 'rbf'")


def default_gamma(X: np.ndarray) -> float:
    """По умолчанию γ = 1/(d · var(X))"""
    d = X.shape[1]
    var = X.var()
    if var <= 0:
        return 1.0
    return 1.0 / (d * var)
