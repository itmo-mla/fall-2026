from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold


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


def build_linear_classifier(X, y, lambdas, C, eps=1e-6):
    w = np.dot(lambdas * y, X)
    w0 = calculate_threshold(X, y, lambdas, C, linear_kernel, eps)
    return w, w0


def calculate_threshold(X, y, lambdas, C, kernel, eps=1e-6):
    scores = np.dot(kernel(X, X), lambdas * y)
    boundary = (lambdas > eps) & (lambdas < C - eps)
    if np.any(boundary):
        return np.mean(scores[boundary] - y[boundary])

    lower = ((y == -1) & (lambdas <= eps)) | ((y == 1) & (lambdas >= C - eps))
    upper = ((y == 1) & (lambdas <= eps)) | ((y == -1) & (lambdas >= C - eps))
    low = np.max(scores[lower] - y[lower])
    high = np.min(scores[upper] - y[upper])
    if low > high + 1e-5:
        raise RuntimeError("No threshold satisfying KKT conditions")
    return (low + high) / 2


def select_parameters(X, y, kernel, C_values, gamma_values=(None,), n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=30)
    folds = []
    for train, valid in cv.split(X, y):
        mean = np.mean(X[train], axis=0)
        std = np.std(X[train], axis=0)
        std[std == 0] = 1
        folds.append(
            ((X[train] - mean) / std, y[train], (X[valid] - mean) / std, y[valid])
        )

    results = []
    for C in C_values:
        for gamma in gamma_values:
            current_kernel = kernel if gamma is None else partial(kernel, gamma=gamma)
            scores = []
            for X_train, y_train, X_valid, y_valid in folds:
                lambdas, _ = solve_dual(X_train, y_train, C, current_kernel)
                w0 = calculate_threshold(X_train, y_train, lambdas, C, current_kernel)
                values = (
                    np.dot(current_kernel(X_valid, X_train), lambdas * y_train) - w0
                )
                predictions = np.where(values >= 0, 1, -1)
                scores.append(np.mean(predictions == y_valid))
            results.append({"C": C, "gamma": gamma, "cv_accuracy": np.mean(scores)})

    best = max(results, key=lambda row: round(row["cv_accuracy"], 12))
    return best, results
