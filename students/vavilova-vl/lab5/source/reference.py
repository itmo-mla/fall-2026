"""Сравнение собственной реализации с sklearn.linear_model.LogisticRegression."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from model import LogisticRegression

# C = ∞ — без регуляризации, как в лекции: по умолчанию sklearn добавляет L2-штраф с C = 1.
# (Раньше это задавалось penalty=None; в sklearn ≥ 1.8 этот параметр объявлен устаревшим.)
REFERENCE_SOLVERS = ("newton-cholesky", "lbfgs")


@dataclass
class Comparison:
    solver: str
    n_iter: int
    weights: float          # max |w_ours − w_sk|, включая свободный член
    proba: float            # max |p_ours − p_sk| на тесте
    disagreements: int      # число объектов теста с разными предсказанными классами


def fit_reference(X: np.ndarray, y01: np.ndarray, solver: str) -> SklearnLogisticRegression:
    return SklearnLogisticRegression(C=np.inf, solver=solver, tol=1e-12, max_iter=10_000).fit(X, y01)


def compare(ours: LogisticRegression, X_train, y_train, X_test, solver: str) -> Comparison:
    ref = fit_reference(X_train, y_train, solver)
    w_ref = np.concatenate([ref.intercept_, ref.coef_.ravel()])
    return Comparison(
        solver=solver,
        n_iter=int(ref.n_iter_[0]),
        weights=float(np.max(np.abs(ours.weights - w_ref))),
        proba=float(np.max(np.abs(ours.predict_proba(X_test) - ref.predict_proba(X_test)[:, 1]))),
        disagreements=int(np.sum(ours.predict(X_test) != ref.predict(X_test))),
    )


def irls_identity(F: np.ndarray, y_pm: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Слайд 8: F̃ᵀF̃ = Q″(w) и F̃ᵀỹ = −Q′(w), поэтому шаг IRLS совпадает с шагом Ньютона.

    Возвращает относительные расхождения обоих равенств.
    """
    from losses import gradient, hessian, margins, sigmoid

    M = margins(F, y_pm, w)
    s = sigmoid(M)
    F_tilde = np.sqrt((1 - s) * s)[:, None] * F
    y_tilde = y_pm * np.exp(-0.5 * M)
    H, g = hessian(F, y_pm, w), gradient(F, y_pm, w)
    return (float(np.max(np.abs(F_tilde.T @ F_tilde - H)) / np.max(np.abs(H))),
            float(np.max(np.abs(F_tilde.T @ y_tilde + g)) / np.max(np.abs(g))))
