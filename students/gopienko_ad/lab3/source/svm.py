import numpy as np
from scipy.optimize import minimize


class SVMClassifier:
    def __init__(
            self,
            C: float = 1.0,
            kernel: str = "linear",
            degree: int = 3,
            gamma: float | str = "scale",
            coef0: float = 1.0,
            tol: float = 1e-6,
            support_tol: float = 1e-5,
            max_iter: int = 1000
    ):
        if C <= 0:
            raise ValueError("C must be positive.")
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        if kernel not in {"linear", "poly", "rbf"}:
            raise ValueError("kernel must be 'linear', 'poly' or 'rbf'.")

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.support_tol = support_tol
        self.max_iter = max_iter

        self.classes_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.y_signed_: np.ndarray | None = None

    def _kernel_matrix(
            self,
            x: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:

        if self.kernel == "linear":
            return x @ z.T

        elif self.kernel == "poly":
            return (self.gamma * (x @ z.T) + self.coef0) ** self.degree

        x_squared = np.sum(
            x * x,
            axis=1,
            keepdims=True
        )

        z_squared = np.sum(
            z * z,
            axis=1,
            keepdims=True
        ).T

        squared_distances = x_squared + z_squared - 2.0 * (x @ z.T)
        squared_distances = np.maximum(squared_distances, 0.0)

        return np.exp(-self.gamma * squared_distances)


    def fit(self, x: np.ndarray, y: np.ndarray) -> "SVMClassifier":

        classes = np.unique(y)

        self.classes_ = classes
        self.X_train_ = x
        self.y_train_ = y
        self.y_signed_ = np.where(y == classes[0], -1.0, 1.0)

        K = self._kernel_matrix(x, x)
        Q = np.outer(self.y_signed_, self.y_signed_) * K

        n_samples = len(x)

        def objective(alpha: np.ndarray) -> float:
            return float(0.5 * alpha @ Q @ alpha - np.sum(alpha))

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return Q @ alpha - np.ones_like(alpha)

        constraints = {
            "type": "eq",
            "fun": lambda alpha: float(alpha @ self.y_signed_),
            "jac": lambda alpha: self.y_signed_
        }

        bounds = [(0.0, self.C)] * n_samples
        alpha0 = np.zeros(n_samples, dtype=float)

        result = minimize(
            objective,
            alpha0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "disp": False,
            },
        )

        self.optimization_result_ = result

        if not result.success:
            raise RuntimeError(
                "SVM dual optimization failed: "
                f"{result.message}"
            )

        alpha = np.clip(np.asarray(result.x, dtype=float),0.0, self.C)

        support_mask = alpha > self.support_tol
        support_indices = np.flatnonzero(support_mask)

        if len(support_indices) == 0:
            raise RuntimeError("No support vectors were found.")

        self.alpha_ = alpha
        self.support_ = support_indices
        self.support_vectors_ = x[support_indices]
        self.support_alpha_ = alpha[support_indices]
        self.support_y_ = self.y_signed_[support_indices]

        weighted_support = self.support_alpha_ * self.support_y_

        margin_mask = (alpha > self.support_tol) & (alpha < self.C - self.support_tol)
        margin_indices = np.flatnonzero(margin_mask)

        if len(margin_indices) > 0:
            bias_indices = margin_indices
        else:
            bias_indices = support_indices

        K_bias = self._kernel_matrix(x[bias_indices], self.support_vectors_,)
        raw_without_bias = K_bias @ weighted_support
        b_values = self.y_signed_[bias_indices] - raw_without_bias

        self.b_ = float(np.mean(b_values))

        if self.kernel == "linear":
            self.w_ = x.T @ (alpha * self.y_signed_)
        else:
            self.w_ = None

        return self

    def decision_function(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("X must be a 2D array.")

        K = self._kernel_matrix(x, self.support_vectors_)

        scores = K @ (self.support_alpha_ * self.support_y_) + self.b_

        return np.asarray(scores, dtype=float)

    def predict(self, x: np.ndarray,) -> np.ndarray:
        scores = self.decision_function(x)

        signed_prediction = np.where(
            scores >= 0.0,
            1.0,
            -1.0,
        )

        return np.where(
            signed_prediction < 0,
            self.classes_[0],
            self.classes_[1],
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        scores = self.decision_function(x)
        clipped = np.clip(scores, -50.0,50.0)
        positive = 1.0 / (1.0 + np.exp(-clipped))
        negative = 1.0 - positive

        return np.column_stack([negative, positive])

    @property
    def n_support_(self) -> np.ndarray:

        counts = [
            np.sum(self.support_y_ < 0),
            np.sum(self.support_y_ > 0),
        ]

        return np.asarray(counts, dtype=int)
