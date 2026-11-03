"""Ручная логистическая регрессия методами Ньютона и IRLS"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticResult:
    """Результат итерационного обучения"""

    weights: np.ndarray
    loss_history: np.ndarray
    iterations: int
    converged: bool


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Добавляет постоянный признак для свободного члена"""
    return np.column_stack((np.ones(len(X)), X))


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Вычисляет sigmoid без переполнения экспоненты"""
    result = np.empty_like(values, dtype=np.float64)
    non_negative = values >= 0

    # Устойчивая форма sigmoid защищает от переполнения
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exponent = np.exp(values[~non_negative])
    result[~non_negative] = exponent / (1.0 + exponent)
    return result


def logistic_loss(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Вычисляет среднюю логистическую ошибку по значениям до sigmoid"""
    scores = X @ weights
    return float(np.mean(np.logaddexp(0.0, scores) - y * scores))


def _solve_linear_system(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Решает систему и обрабатывает вырожденную матрицу"""
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> LogisticResult:
    """Обучает логистическую регрессию методом Ньютона-Рафсона"""
    weights = np.zeros(X.shape[1], dtype=np.float64)
    loss_history = []
    converged = False
    sample_count = len(y)

    for _ in range(max_iter):
        probabilities = stable_sigmoid(X @ weights)
        # Диагональные веса зависят от уверенности модели
        diagonal_weights = probabilities * (1.0 - probabilities)
        diagonal_weights = np.clip(diagonal_weights, 1e-12, None)

        gradient = X.T @ (probabilities - y) / sample_count
        hessian = X.T @ (X * diagonal_weights[:, None]) / sample_count
        step = _solve_linear_system(hessian, gradient)
        updated_weights = weights - step
        loss_history.append(logistic_loss(X, y, updated_weights))

        if np.linalg.norm(updated_weights - weights) < tolerance:
            weights = updated_weights
            converged = True
            break
        weights = updated_weights

    return LogisticResult(
        weights=weights,
        loss_history=np.asarray(loss_history),
        iterations=len(loss_history),
        converged=converged,
    )


def fit_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> LogisticResult:
    """Обучает логистическую регрессию методом IRLS"""
    weights = np.zeros(X.shape[1], dtype=np.float64)
    loss_history = []
    converged = False
    sample_count = len(y)

    for _ in range(max_iter):
        scores = X @ weights
        probabilities = stable_sigmoid(scores)
        diagonal_weights = probabilities * (1.0 - probabilities)
        # Ограничиваем веса снизу, чтобы избежать деления на ноль
        diagonal_weights = np.clip(diagonal_weights, 1e-12, None)

        working_response = scores + (y - probabilities) / diagonal_weights
        normal_matrix = X.T @ (X * diagonal_weights[:, None]) / sample_count
        right_side = X.T @ (diagonal_weights * working_response) / sample_count
        updated_weights = _solve_linear_system(normal_matrix, right_side)
        loss_history.append(logistic_loss(X, y, updated_weights))

        if np.linalg.norm(updated_weights - weights) < tolerance:
            weights = updated_weights
            converged = True
            break
        weights = updated_weights

    return LogisticResult(
        weights=weights,
        loss_history=np.asarray(loss_history),
        iterations=len(loss_history),
        converged=converged,
    )


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Возвращает вероятность положительного класса"""
    return stable_sigmoid(X @ weights)


def predict(
    X: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Преобразует вероятности в номера классов"""
    return (predict_proba(X, weights) >= threshold).astype(np.int64)
