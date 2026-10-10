"""SVM через двойственную задачу по множителям Лагранжа.

Метки y_i в {-1, +1}. Минимизируется

    f(λ) = -1^T λ + (1/2) λ^T Q λ,   Q_ij = y_i y_j K(x_i, x_j)

при ограничениях

    λ^T y = 0,   0 ≤ λ_i ≤ C.

Решающее правило:

    a(x) = sign( sum_i λ_i y_i K(x, x_i) - w_0 ).

Для опорного на границе полосы (0 < λ_i < C)

    w_0 = sum_j λ_j y_j K(x_j, x_i) - y_i.

В sklearn свободный член intercept_ равен -w_0, а dual_coef_ хранит λ_i y_i
только на опорных векторах.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


SV_TOL = 1e-5


def kernel_matrix(
    left: np.ndarray,
    right: np.ndarray,
    kind: str,
    degree: int = 3,
    gamma: float = 0.5,
) -> np.ndarray:
    """K(left, right). Линейное, полиномиальное (⟨x, x'⟩ + 1)^d и гауссово."""
    gram = left @ right.T
    if kind == "linear":
        return gram
    if kind == "poly":
        return (gram + 1.0) ** degree
    if kind == "rbf":
        left_sq = np.sum(left * left, axis=1)[:, None]
        right_sq = np.sum(right * right, axis=1)[None, :]
        dist2 = np.maximum(left_sq + right_sq - 2.0 * gram, 0.0)
        return np.exp(-gamma * dist2)
    raise ValueError(kind)


@dataclass
class SVMModel:
    kernel: str
    C: float
    degree: int
    gamma: float
    X: np.ndarray
    y: np.ndarray
    lam: np.ndarray
    w0: float
    support: np.ndarray
    boundary: np.ndarray
    w: np.ndarray | None
    objective: float
    constraint: float
    solver: str

    def scores(self, X_query: np.ndarray) -> np.ndarray:
        gram = kernel_matrix(X_query, self.X, self.kernel, self.degree, self.gamma)
        return gram @ (self.lam * self.y) - self.w0

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        signed = np.sign(self.scores(X_query))
        signed[signed == 0.0] = 1.0
        return signed


def _objective_and_grad(Q: np.ndarray):
    ones = np.ones(Q.shape[0], dtype=np.float64)

    def objective(lam: np.ndarray) -> float:
        return float(-lam.sum() + 0.5 * lam @ Q @ lam)

    def grad(lam: np.ndarray) -> np.ndarray:
        return -ones + Q @ lam

    return objective, grad


def _solve_dual(Q: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, float, float, str]:
    """Минимум f(λ). Сначала SLSQP, как в постановке с equality constraint.

    Если равенство λ^T y = 0 не выполнено, берётся trust-constr с той же f.
    """
    n = len(y)
    objective, grad = _objective_and_grad(Q)
    lam0 = np.zeros(n, dtype=np.float64)
    slsqp = minimize(
        objective,
        lam0,
        method="SLSQP",
        jac=grad,
        bounds=[(0.0, C)] * n,
        constraints={"type": "eq", "fun": lambda lam: float(lam @ y), "jac": lambda lam: y},
        options={"maxiter": 800, "ftol": 1e-12, "disp": False},
    )
    lam = np.clip(slsqp.x, 0.0, C)
    residual = float(lam @ y)
    solver = "SLSQP"
    if abs(residual) > 1e-6 or not slsqp.success:
        other = minimize(
            objective,
            lam0,
            method="trust-constr",
            jac=grad,
            hess=lambda lam: Q,
            bounds=Bounds(0.0, C),
            constraints=LinearConstraint(y.reshape(1, -1), 0.0, 0.0),
            options={"maxiter": 400, "gtol": 1e-10, "verbose": 0},
        )
        lam_other = np.clip(other.x, 0.0, C)
        residual_other = float(lam_other @ y)
        if abs(residual_other) < abs(residual):
            lam = lam_other
            residual = residual_other
            solver = "trust-constr"
    if abs(residual) > 1e-4:
        raise RuntimeError(f"ограничение λ^T y = 0 не выполнено, остаток {residual}")
    return lam, objective(lam), residual, solver


def _bias(K: np.ndarray, y: np.ndarray, lam: np.ndarray, C: float) -> tuple[float, np.ndarray, np.ndarray]:
    """w_0 — среднее по опорным на границе полосы. Если таких нет, по всем опорным."""
    support = np.flatnonzero(lam > SV_TOL)
    boundary = np.flatnonzero((lam > SV_TOL) & (lam < C - SV_TOL))
    used = boundary if len(boundary) else support
    if len(used) == 0:
        raise RuntimeError("опорных векторов нет: C слишком мал или классы неразличимы")
    # sum_j λ_j y_j K(x_j, x_i) для выбранных i, затем минус y_i
    raw = K[used] @ (lam * y) - y[used]
    return float(np.mean(raw)), support, boundary


def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "linear",
    C: float = 1.0,
    degree: int = 3,
    gamma: float = 0.5,
) -> SVMModel:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if set(np.unique(y).tolist()) - {-1.0, 1.0}:
        raise ValueError("метки SVM должны быть -1 и +1")
    if C <= 0:
        raise ValueError("C должен быть положительным")

    K = kernel_matrix(X, X, kernel, degree=degree, gamma=gamma)
    Q = np.outer(y, y) * K
    Q = 0.5 * (Q + Q.T)
    lam, value, residual, solver = _solve_dual(Q, y, C)
    w0, support, boundary = _bias(K, y, lam, C)
    weights = (lam * y) @ X if kernel == "linear" else None
    return SVMModel(
        kernel=kernel,
        C=float(C),
        degree=int(degree),
        gamma=float(gamma),
        X=X,
        y=y,
        lam=lam,
        w0=w0,
        support=support,
        boundary=boundary,
        w=None if weights is None else np.asarray(weights, dtype=np.float64),
        objective=value,
        constraint=residual,
        solver=solver,
    )
