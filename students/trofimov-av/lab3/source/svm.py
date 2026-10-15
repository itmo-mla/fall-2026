import numpy as np
from scipy.optimize import minimize

from kernels import kernel_matrix


def solve_dual(x, y, c=1.0, kernel="linear", gamma=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if c <= 0:
        raise ValueError("Параметр c должен быть положительным")
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("Ожидаются матрица признаков и вектор меток той же длины")
    if not np.array_equal(np.unique(y), [-1, 1]):
        raise ValueError("Метки двух классов должны быть -1 и 1")

    q = np.outer(y, y) * kernel_matrix(x, x, kernel, gamma)

    def objective(alpha):
        return 0.5 * alpha @ q @ alpha - alpha.sum()

    def gradient(alpha):
        return q @ alpha - np.ones(len(y))

    result = minimize(
        objective,
        np.zeros(len(y)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, c)] * len(y),
        constraints={"type": "eq", "fun": lambda alpha: alpha @ y, "jac": lambda alpha: y},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Не удалось решить двойственную задачу: {result.message}")
    return result.x


class LinearSVM:
    def __init__(self, c=1.0):
        self.c = c

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.alpha = solve_dual(x, y, c=self.c)
        self.weights = (self.alpha * y) @ x

        tolerance = min(1e-6, self.c * 1e-3)
        self.support = np.flatnonzero(self.alpha > tolerance)
        free = (self.alpha > tolerance) & (self.alpha < self.c - tolerance)
        residuals = y - x @ self.weights

        if np.any(free):
            self.bias = residuals[free].mean()
        else:
            lower = ((y == 1) & (self.alpha < self.c - tolerance)) | (
                (y == -1) & (self.alpha > tolerance)
            )
            upper = ((y == -1) & (self.alpha < self.c - tolerance)) | (
                (y == 1) & (self.alpha > tolerance)
            )
            self.bias = (residuals[lower].max() + residuals[upper].min()) / 2
        return self

    def decision_function(self, x):
        return np.asarray(x, dtype=float) @ self.weights + self.bias

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0, 1, -1)
