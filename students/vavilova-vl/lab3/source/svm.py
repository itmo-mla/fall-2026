from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def linear_kernel() -> Kernel:
    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B.T

    return kernel


def polynomial_kernel(degree: int, gamma: float, coef0: float) -> Kernel:
    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (gamma * (A @ B.T) + coef0) ** degree

    return kernel


def rbf_kernel(gamma: float) -> Kernel:
    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(A**2, axis=1)[:, None]
            + np.sum(B**2, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        return np.exp(-gamma * np.maximum(squared, 0.0))

    return kernel


class DualSVM:
    """SVM с мягким зазором: двойственная задача по лямбда решается scipy.optimize.minimize.

    min_lambda  1/2 * lambda^T Q lambda - sum(lambda),   Q_ij = y_i y_j K(x_i, x_j)
    при условиях 0 <= lambda_i <= C,  sum(lambda_i y_i) = 0.

    Классификатор: a(x) = sign(sum(lambda_i y_i K(x, x_i)) - w0).
    """

    def __init__(self, kernel: Kernel, C: float = 1.0, ftol: float = 1e-10, max_iter: int = 2000):
        self.kernel = kernel
        self.C = C
        self.ftol = ftol
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DualSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Одинаковые объекты одного класса входят в задачу только через сумму своих лямбд,
        # поэтому группа из m дубликатов заменяется одной переменной с границей m * C.
        unique, inverse, counts = np.unique(
            np.column_stack([X, y]), axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()
        X_unique, y_unique = unique[:, :-1], unique[:, -1]
        upper = self.C * counts

        Q = (y_unique[:, None] * y_unique[None, :]) * self.kernel(X_unique, X_unique)

        def objective(lambdas: np.ndarray) -> float:
            return 0.5 * lambdas @ Q @ lambdas - lambdas.sum()

        def gradient(lambdas: np.ndarray) -> np.ndarray:
            return Q @ lambdas - 1.0

        result = minimize(
            objective,
            x0=np.zeros(len(y_unique)),
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(np.zeros(len(y_unique)), upper)),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda lambdas: lambdas @ y_unique,
                    "jac": lambda lambdas: y_unique,
                }
            ],
            options={"ftol": self.ftol, "maxiter": self.max_iter},
        )

        # Лямбда группы делится поровну между её дубликатами.
        lambdas = np.clip(result.x, 0.0, upper)[inverse] / counts[inverse]
        eps = 1e-8 * self.C
        lambdas[lambdas < eps] = 0.0
        lambdas[lambdas > self.C - eps] = self.C

        support = lambdas > 0.0
        on_margin = support & (lambdas < self.C)

        self.lambdas_ = lambdas
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.dual_coef_ = lambdas[support] * y[support]
        self.n_margin_ = int(on_margin.sum())
        self.n_iter_ = int(result.nit)
        self.message_ = str(result.message)

        # w0 восстанавливается по опорным граничным объектам (0 < lambda < C): для них M_i = 1,
        # откуда w0 = <w, x_i> - y_i. Если таких нет - берётся середина допустимого отрезка.
        scores_without_w0 = self.kernel(X, self.support_vectors_) @ self.dual_coef_
        candidates = scores_without_w0 - y
        if on_margin.any():
            self.w0_ = float(np.mean(candidates[on_margin]))
        else:
            lower = ((y > 0) & (lambdas < self.C)) | ((y < 0) & (lambdas > 0))
            upper_side = ((y > 0) & (lambdas > 0)) | ((y < 0) & (lambdas < self.C))
            self.w0_ = float((candidates[lower].max() + candidates[upper_side].min()) / 2.0)

        self.dual_objective_ = float(
            0.5 * self.dual_coef_ @ scores_without_w0[support] - lambdas.sum()
        )
        self.kkt_violation_ = self._kkt_violation(y * (scores_without_w0 - self.w0_), y)
        return self

    def _kkt_violation(self, margins: np.ndarray, y: np.ndarray) -> float:
        """Наибольшее нарушение условий Каруша-Куна-Таккера (0 в точном оптимуме).

        lambda_i = 0      =>  M_i >= 1   (периферийный)
        0 < lambda_i < C  =>  M_i  = 1   (опорный-граничный)
        lambda_i = C      =>  M_i <= 1   (опорный-нарушитель)
        """
        lambdas = self.lambdas_
        violation = np.abs(margins - 1.0)
        violation[lambdas == 0.0] = np.maximum(0.0, 1.0 - margins[lambdas == 0.0])
        violation[lambdas == self.C] = np.maximum(0.0, margins[lambdas == self.C] - 1.0)
        return float(max(violation.max(), abs(lambdas @ y)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel(np.asarray(X, dtype=float), self.support_vectors_)
        return K @ self.dual_coef_ - self.w0_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def linear_weights(self) -> np.ndarray:
        """w = sum(lambda_i y_i x_i). Имеет смысл только для линейного ядра."""
        return self.dual_coef_ @ self.support_vectors_
