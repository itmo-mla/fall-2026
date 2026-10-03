import numpy as np
from scipy.optimize import minimize

from kernels import linear_kernel, rbf_kernel


class DualSVM:
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, tolerance=1e-6):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tolerance = tolerance

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)

        K = self._kernel(self.X_train, self.X_train)
        yyK = np.outer(self.y_train, self.y_train) * K
        n_objects = len(self.X_train)

        def objective(lambdas):
            return 0.5 * lambdas @ yyK @ lambdas - np.sum(lambdas)

        def gradient(lambdas):
            return yyK @ lambdas - np.ones_like(lambdas)

        constraints = {
            "type": "eq",
            "fun": lambda lambdas: lambdas @ self.y_train,
            "jac": lambda lambdas: self.y_train,
        }
        bounds = [(0.0, self.C) for _ in range(n_objects)]
        initial = np.zeros(n_objects, dtype=float)

        result = minimize(
            objective,
            initial,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-9}
        )

        if not result.success:
            raise RuntimeError(result.message)

        self.lambdas = result.x
        self.support_mask = self.lambdas > self.tolerance
        self.margin_mask = (
            (self.lambdas > self.tolerance)
            & (self.lambdas < self.C - self.tolerance)
        )

        self.support_vectors = self.X_train[self.support_mask]
        self.support_labels = self.y_train[self.support_mask]
        self.support_lambdas = self.lambdas[self.support_mask]

        self.w = None
        if self.kernel == "linear":
            self.w = (self.lambdas * self.y_train) @ self.X_train

        self.w0 = self._compute_w0(K)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = self._kernel(X, self.support_vectors)
        return K @ (self.support_lambdas * self.support_labels) - self.w0

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)

    def _kernel(self, X, Y):
        if self.kernel == "linear":
            return linear_kernel(X, Y)
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma)
        raise ValueError("Неизвестное ядро")

    def _compute_w0(self, K):
        if np.any(self.margin_mask):
            indices = np.flatnonzero(self.margin_mask)
        else:
            indices = np.flatnonzero(self.support_mask)

        values = []
        for index in indices:
            decision_without_w0 = np.sum(
                self.lambdas
                * self.y_train
                * K[index]
            )
            values.append(decision_without_w0 - self.y_train[index])

        return float(np.mean(values))

    def margins(self, X, y):
        return np.asarray(y, dtype=float) * self.decision_function(X)
