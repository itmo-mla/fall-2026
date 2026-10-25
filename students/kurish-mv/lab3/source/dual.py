import numpy as np
from scipy.optimize import minimize

from kernel import compute_kernel


def dual_objective(lambdas: np.ndarray, Q: np.ndarray) -> float:
    """Минимизируемая форма: ½λᵀQλ − Σλᵢ (двойственная с обратным знаком)"""
    return 0.5 * lambdas @ Q @ lambdas - np.sum(lambdas)


def dual_gradient(lambdas: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Градиент минимизируемой формы: Qλ − 1"""
    return Q @ lambdas - np.ones_like(lambdas)


def equality_constraint(lambdas: np.ndarray, y: np.ndarray) -> float:
    """Ограничение-равенство Σλᵢyᵢ = 0"""
    return float(lambdas @ y)


def equality_gradient(lambdas: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Градиент ограничения равенства совпадает с вектором меток"""
    return y.copy()


def build_q(X: np.ndarray, y: np.ndarray, kind: str,
            gamma: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Матрица Q = (y yᵀ) ∘ K и сама матрица ядра на обучающей выборке"""
    K = compute_kernel(X, X, kind=kind, gamma=gamma)
    return np.outer(y, y) * K, K


def solve_dual(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    kind: str = "linear",
    gamma: float | None = None,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Решение двойственной задачи SVM; возвращает λ и матрицу K.

    x0 задаёт начальную точку: тёплый старт ускоряет перебор по C.
    """
    if C <= 0:
        raise ValueError("Параметр C должен быть положительным")

    Q, K = build_q(X, y, kind=kind, gamma=gamma)
    n = len(y)
    start = np.zeros(n) if x0 is None else np.clip(np.asarray(x0, float), 0.0, C)

    constraint = {
        "type": "eq",
        "fun": equality_constraint,
        "jac": equality_gradient,
        "args": (y,),
    }
    result = minimize(
        fun=dual_objective,
        x0=start,
        args=(Q,),
        jac=dual_gradient,
        method="SLSQP",
        bounds=[(0.0, C)] * n,
        constraints=[constraint],
        options={"maxiter": 2000, "ftol": 1e-9, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP не сошёлся: {result.message}")
    return result.x, K


def gradient_mismatch(
    lambdas: np.ndarray, Q: np.ndarray, eps: float = 1e-7
) -> float:
    """Максимальное расхождение аналитического и численного градиентов"""
    analytic = dual_gradient(lambdas, Q)
    numerical = np.empty_like(lambdas)
    for i in range(len(lambdas)):
        step = np.zeros_like(lambdas)
        step[i] = eps
        plus = dual_objective(lambdas + step, Q)
        minus = dual_objective(lambdas - step, Q)
        numerical[i] = (plus - minus) / (2.0 * eps)
    return float(np.max(np.abs(analytic - numerical)))
