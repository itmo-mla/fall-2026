import numpy as np
from scipy.optimize import minimize


def linear_kernel(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return X @ Z.T


def rbf_kernel(
    X: np.ndarray,
    Z: np.ndarray,
    gamma: float,
) -> np.ndarray:
    distances = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(Z**2, axis=1)[None, :]
        - 2 * X @ Z.T
    )
    return np.exp(-gamma * np.maximum(distances, 0))


def kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    kernel: str,
    gamma: float,
) -> np.ndarray:
    if kernel == "linear":
        return linear_kernel(X, Z)
    if kernel == "rbf":
        return rbf_kernel(X, Z, gamma)
    raise ValueError(f"Неизвестное ядро: {kernel}")


def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float = 1.0,
    gamma: float = 1.0,
) -> dict:
    K = kernel_matrix(X, X, kernel, gamma)
    Q = (y[:, None] * y[None, :]) * K

    def objective(alpha: np.ndarray) -> float:
        return float(0.5 * alpha @ Q @ alpha - np.sum(alpha))

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return Q @ alpha - np.ones_like(alpha)

    # В двойственной задаче сумма alpha_i * y_i должна быть равна нулю
    constraint = {
        "type": "eq",
        "fun": lambda alpha: alpha @ y,
        "jac": lambda alpha: y,
    }
    result = minimize(
        objective,
        np.zeros(len(X)),
        jac=gradient,
        bounds=[(0.0, C)] * len(X),
        constraints=constraint,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"Оптимизация не сошлась: {result.message}")

    tolerance = 1e-6
    alpha = np.clip(result.x, 0, C)
    support_mask = alpha > tolerance
    margin_mask = support_mask & (alpha < C - tolerance)

    # Свободный член усредняется по опорным объектам внутри границы C
    bias_mask = margin_mask if np.any(margin_mask) else support_mask
    coefficients = alpha[support_mask] * y[support_mask]
    bias = np.mean(
        y[bias_mask] - K[np.ix_(bias_mask, support_mask)] @ coefficients
    )
    weights = (
        X[support_mask].T @ coefficients if kernel == "linear" else None
    )

    return {
        "alpha": alpha[support_mask],
        "support_vectors": X[support_mask],
        "support_labels": y[support_mask],
        "bias": float(bias),
        "kernel": kernel,
        "gamma": gamma,
        "weights": weights,
        "constraint_error": float(abs(alpha @ y)),
    }


def decision_function(model: dict, X: np.ndarray) -> np.ndarray:
    if model["kernel"] == "linear":
        return X @ model["weights"] + model["bias"]

    K = kernel_matrix(
        X,
        model["support_vectors"],
        model["kernel"],
        model["gamma"],
    )
    coefficients = model["alpha"] * model["support_labels"]
    return K @ coefficients + model["bias"]


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    scores = decision_function(model, X)
    return np.where(scores >= 0, 1, -1)
