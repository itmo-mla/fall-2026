import numpy as np
from scipy.optimize import minimize


def solve_dual(X, y, C=1.0):
    if C <= 0:
        raise ValueError("C must be positive")

    Q = np.outer(y, y) * np.dot(X, X.T)

    def objective(lambdas):
        return 0.5 * np.dot(lambdas, np.dot(Q, lambdas)) - np.sum(lambdas)

    def gradient(lambdas):
        return np.dot(Q, lambdas) - 1

    result = minimize(
        objective,
        np.zeros(len(y)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, C)] * len(y),
        constraints={
            "type": "eq",
            "fun": lambda lambdas: np.dot(lambdas, y),
            "jac": lambda lambdas: y,
        },
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"Dual optimization failed: {result.message}")

    return result.x, result.fun
