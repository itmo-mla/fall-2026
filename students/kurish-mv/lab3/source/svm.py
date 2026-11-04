import numpy as np

from dual import solve_dual
from kernel import compute_kernel, default_gamma
from dataclasses import dataclass


class SVM:
    """SVM с решением двойственной задачи по λ методом SLSQP"""

    def __init__(
        self,
        C: float = 1.0,
        kind: str = "linear",
        gamma: float | None = None,
        tol: float = 1e-6,
    ) -> None:
        if C <= 0:
            raise ValueError("Параметр C должен быть положительным")
        if kind not in ("linear", "rbf"):
            raise ValueError("Поддерживаются ядра 'linear' и 'rbf'")
        if tol <= 0:
            raise ValueError("Порог tol должен быть положительным")
        self.C = float(C)
        self.kind = kind
        self.gamma = gamma
        self.tol = float(tol)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._lambdas: np.ndarray | None = None
        self._kernel_matrix: np.ndarray | None = None
        self._support_indices: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray,
            x0: np.ndarray | None = None) -> "SVM":
        """Решает двойственную задачу и восстанавливает b по условиям KKT.

        x0 — необязательный тёплый старт по λ (используется в переборе C).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError("Ожидаются X размера (n, d) и y длины n")
        if not np.isin(y, [-1.0, 1.0]).all():
            raise ValueError("Метки должны быть из множества {-1, +1}")
        if len(np.unique(y)) < 2:
            raise ValueError("В обучающей выборке должны быть оба класса")

        if self.gamma is None and self.kind == "rbf":
            self.gamma = default_gamma(X)

        lambdas, K = solve_dual(
            X, y, C=self.C, kind=self.kind, gamma=self.gamma, x0=x0
        )
        self._X_train, self._y_train = X, y
        self._lambdas, self._kernel_matrix = lambdas, K
        self._support_indices = np.flatnonzero(lambdas > self.tol)
        if self._support_indices.size == 0:
            raise RuntimeError("Оптимизатор не выделил опорные векторы")

        scores = K @ (lambdas * y)
        free = np.flatnonzero((lambdas > self.tol) & (lambdas < self.C - self.tol))
        if free.size:
            self.b_ = float(np.mean(y[free] - scores[free]))
        else:
            self.b_ = self._intercept_from_bounds(y, scores)

        self.w_ = (lambdas * y) @ X if self.kind == "linear" else None
        return self

    def _intercept_from_bounds(
        self, y: np.ndarray, scores: np.ndarray
    ) -> float:
        """Середина допустимого интервала для b, если свободных ОВ нет"""
        lower, upper = [], []
        for lam, target, score in zip(self._lambdas, y, scores):
            bound = target - score
            if lam <= self.tol:
                (lower if target > 0 else upper).append(bound)
            elif lam >= self.C - self.tol:
                (upper if target > 0 else lower).append(bound)
        if not lower or not upper:
            raise RuntimeError("Не удалось определить b по условиям KKT")
        return float((max(lower) + min(upper)) / 2.0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) = wᵀx + b для линейного ядра или ΣλᵢyᵢK(xᵢ,x) + b для RBF"""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.kind == "linear":
            return X @ self.w_ + self.b_
        support = self._X_train[self._support_indices]
        dual_coef = self._lambdas[self._support_indices] * self._y_train[
            self._support_indices
        ]
        kernel = compute_kernel(support, X, kind=self.kind, gamma=self.gamma)
        return dual_coef @ kernel + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Класс по знаку функции решения; ноль относится к +1"""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def _check_fitted(self) -> None:
        if self._lambdas is None or self._X_train is None:
            raise RuntimeError("Сначала вызовите fit()")

    @property
    def support_vectors(self) -> np.ndarray:
        self._check_fitted()
        return self._X_train[self._support_indices]

    @property
    def support_indices(self) -> np.ndarray:
        self._check_fitted()
        return self._support_indices

    @property
    def n_support(self) -> int:
        return 0 if self._support_indices is None else int(
            self._support_indices.size
        )

    @property
    def lambdas(self) -> np.ndarray:
        self._check_fitted()
        return self._lambdas

    @property
    def kernel_matrix(self) -> np.ndarray:
        self._check_fitted()
        return self._kernel_matrix

@dataclass
class SVMResult:
    """Метрики и предсказания одной модели SVM"""

    key: str
    label: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_support: int
    train_acc: float
    test_acc: float
    predictions: np.ndarray
