"""Классификатор логистической регрессии с интерфейсом в стиле sklearn."""

import numpy as np

from data import add_bias
from losses import sigmoid
from solvers import irls, irls_glm, newton_raphson

SOLVERS = {"newton": newton_raphson, "irls": irls, "irls_glm": irls_glm}


class LogisticRegression:
    """a(x) = sign(⟨w, x⟩), P(y = 1 | x) = σ(⟨w, x⟩); w_0 — свободный член.

    init="ols" — нулевое приближение из шага 1 IRLS (слайд 9): w = (FᵀF)⁻¹Fᵀy, y ∈ {−1, +1};
    init="zero" — w = 0, то есть σ_i = 1/2 для всех объектов.
    """

    def __init__(self, method: str = "irls", init: str = "ols", tol: float = 1e-10, max_iter: int = 100):
        if method not in SOLVERS:
            raise ValueError(f"method должен быть одним из {list(SOLVERS)}")
        self.method, self.init, self.tol, self.max_iter = method, init, tol, max_iter

    def initial_weights(self, F: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
        if self.init == "zero":
            return np.zeros(F.shape[1])
        if self.init == "ols":
            w, *_ = np.linalg.lstsq(F, y_pm, rcond=None)
            return w
        raise ValueError(f"неизвестная инициализация {self.init!r}")

    def fit(self, X: np.ndarray, y01: np.ndarray, w0: np.ndarray | None = None) -> "LogisticRegression":
        F = add_bias(X)
        y_pm = 2.0 * y01 - 1.0
        w0 = self.initial_weights(F, y_pm) if w0 is None else w0
        target = y01 if self.method == "irls_glm" else y_pm
        w, self.history_ = SOLVERS[self.method](F, target, w0, tol=self.tol, max_iter=self.max_iter)
        self.intercept_, self.coef_ = w[0], w[1:]
        self.n_iter_ = self.history_.n_iter
        return self

    @property
    def weights(self) -> np.ndarray:
        return np.concatenate([[self.intercept_], self.coef_])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y = 1 | x) = σ(⟨w, x⟩) (слайд 20)."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(float)
