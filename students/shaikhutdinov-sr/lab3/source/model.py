import numpy as np

from optimization import solve_dual


def kernel_matrix(
    features: np.ndarray,
    training_features: np.ndarray,
    kernel: str,
    gamma: float,
) -> np.ndarray:
    scalar_products = np.dot(features, training_features.T)
    if kernel == "linear":
        return scalar_products
    if kernel == "rbf":
        squared_distances = (
            np.sum(features ** 2, axis=1)[:, None]
            + np.sum(training_features ** 2, axis=1)[None, :]
            - 2 * scalar_products
        )
        return np.exp(-gamma * np.maximum(squared_distances, 0))
    raise ValueError(f"Неизвестное ядро: {kernel}")


def calculate_intercept(
    multipliers: np.ndarray,
    labels: np.ndarray,
    scores_without_intercept: np.ndarray,
    C: float,
    tolerance: float,
) -> float:
    free_support = (multipliers > tolerance) & (multipliers < C - tolerance)
    if np.any(free_support):
        return float(np.mean(labels[free_support] - scores_without_intercept[free_support]))

    # Без свободных опорных векторов условия KKT задают интервал допустимых b.
    at_zero = multipliers <= tolerance
    at_limit = multipliers >= C - tolerance
    lower_mask = (at_zero & (labels == 1)) | (at_limit & (labels == -1))
    upper_mask = (at_zero & (labels == -1)) | (at_limit & (labels == 1))
    intercept_candidates = labels - scores_without_intercept
    lower_bound = np.max(intercept_candidates[lower_mask])
    upper_bound = np.min(intercept_candidates[upper_mask])
    if lower_bound > upper_bound + 1e-4:
        raise RuntimeError("Решение не задаёт допустимого интервала для смещения")
    return float((lower_bound + upper_bound) / 2)


class SVM:
    def __init__(self, C: float = 1.0, kernel: str = "linear", gamma: float = 0.1):
        if C <= 0 or (kernel == "rbf" and gamma <= 0):
            raise ValueError("C и gamma для RBF должны быть положительными")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tolerance = 1e-6 * min(1.0, C)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SVM":
        gram_matrix = kernel_matrix(features, features, self.kernel, self.gamma)
        result = solve_dual(gram_matrix, labels, self.C, self.tolerance)
        self.multipliers = result.x

        signed_multipliers = self.multipliers * labels
        scores_without_intercept = np.dot(gram_matrix, signed_multipliers)
        self.intercept = calculate_intercept(
            self.multipliers, labels, scores_without_intercept, self.C, self.tolerance
        )
        self.support_indices = np.flatnonzero(self.multipliers > self.tolerance)
        self.support_features = features[self.support_indices]
        self.support_coefficients = signed_multipliers[self.support_indices]
        self.weights = np.dot(features.T, signed_multipliers) if self.kernel == "linear" else None
        self.dual_value = -float(result.fun)
        self.iterations = result.nit
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return np.dot(features, self.weights) + self.intercept
        similarities = kernel_matrix(features, self.support_features, self.kernel, self.gamma)
        return np.dot(similarities, self.support_coefficients) + self.intercept

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(features) >= 0, 1, -1)
