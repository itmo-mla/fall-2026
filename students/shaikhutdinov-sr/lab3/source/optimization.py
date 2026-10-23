import numpy as np
from scipy.optimize import OptimizeResult, minimize


def check_dual_constraints(
    multipliers: np.ndarray,
    labels: np.ndarray,
    C: float,
    tolerance: float,
) -> None:
    equality_error = abs(np.dot(multipliers, labels))
    lower_error = max(0, -multipliers.min())
    upper_error = max(0, multipliers.max() - C)
    if max(equality_error, lower_error, upper_error) > tolerance:
        raise RuntimeError("Нарушены ограничения двойственной задачи")


def solve_dual(
    gram_matrix: np.ndarray,
    labels: np.ndarray,
    C: float,
    tolerance: float,
) -> OptimizeResult:
    dual_matrix = np.outer(labels, labels) * gram_matrix

    def objective(multipliers: np.ndarray) -> float:
        quadratic_term = np.dot(multipliers, np.dot(dual_matrix, multipliers))
        return float(0.5 * quadratic_term - multipliers.sum())

    def gradient(multipliers: np.ndarray) -> np.ndarray:
        return np.dot(dual_matrix, multipliers) - 1

    result = minimize(
        objective,
        x0=np.zeros(len(labels)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, C)] * len(labels),
        constraints={
            "type": "eq",
            "fun": lambda multipliers: np.dot(multipliers, labels),
            "jac": lambda multipliers: labels.astype(float),
        },
        options={"ftol": 1e-9, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Двойственная задача не решена: {result.message}")
    check_dual_constraints(result.x, labels, C, tolerance)
    return result
