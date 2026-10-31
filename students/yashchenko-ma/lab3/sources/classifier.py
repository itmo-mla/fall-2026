"""Шаги 3-4: SVM-классификатор (линейный и с ядром): опорные векторы, порог b, предсказание."""
import numpy as np
from kernels import linear_kernel, rbf_kernel, poly_kernel
from svm_dual import solve_dual


class DualSVM:
    """SVM с мягким зазором, обучаемый решением двойственной задачи."""

    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3, coef0=1.0, tol=1e-5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        elif self.kernel == 'poly':
            return poly_kernel(X1, X2, self.degree, self.coef0)
        raise ValueError(self.kernel)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        K = self._kernel(X, X)
        alpha, res = solve_dual(K, y, self.C, self.tol)

        self.X_train_ = X
        self.y_train_ = y
        self.alpha_full_ = alpha

        alpha_tol = self.tol * max(1.0, self.C) #  alpha_tol  нужен из-за численных погрешностей оптимизатора

        # sv_mask = alpha > self.tol * self.C # после отбора опорных векторов почти всегда истинна
        sv_mask = alpha > alpha_tol
        free_mask = (
            (alpha > alpha_tol) &
            (alpha < self.C - alpha_tol)
        )

        self.alpha_ = alpha[sv_mask]
        self.sv_X_ = X[sv_mask]
        self.sv_y_ = y[sv_mask]
        self.n_sv_ = sv_mask.sum()

        # b: усреднение по опорным векторам НЕ на границе (0 < alpha < C)
        free_mask = (self.alpha_ > self.tol * self.C) & (self.alpha_ < self.C * (1 - self.tol))
        if free_mask.sum() == 0:
            free_mask = np.ones_like(self.alpha_, dtype=bool)
        K_sv = self._kernel(self.sv_X_[free_mask], self.sv_X_)
        margin = self.sv_y_[free_mask] - (K_sv @ (self.alpha_ * self.sv_y_))
        self.b_ = margin.mean()

        if self.kernel == 'linear':
            self.w_ = (self.alpha_ * self.sv_y_) @ self.sv_X_
        self.opt_result_ = res
        return self

    def decision_function(self, X):
        K = self._kernel(X, self.sv_X_)
        return K @ (self.alpha_ * self.sv_y_) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))
