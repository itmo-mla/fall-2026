import numpy as np


class LinearRegressionSVD:
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionSVD":

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())

        X_centered = X - x_mean
        y_centered = y - y_mean

        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)

        factors = np.divide(
            s,
            s ** 2 + self.alpha,
            out=np.zeros_like(s),
            where=(s ** 2 + self.alpha) > 0
        )

        self.coef_ = vt.T @ (factors * (u.T @ y_centered))
        self.intercept_ = y_mean - float(x_mean @ self.coef_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
