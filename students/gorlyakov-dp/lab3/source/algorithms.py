import numpy as np
from scipy.optimize import minimize


def linear_kernel(X, Z):
    return np.dot(X, Z.T)


def polynomial_kernel(X, Z, degree=3):
    return (np.dot(X, Z.T) + 1) ** degree


def rbf_kernel(X, Z, gamma=1.0):
    squared_distances = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * squared_distances)


def solve_dual(X, y, C=1.0, kernel=linear_kernel):
    if C <= 0:
        raise ValueError("C must be positive")

    Q = np.outer(y, y) * kernel(X, X)

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
