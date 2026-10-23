import numpy as np
from scipy.optimize import minimize

from kernels import make_kernel


class DualSVM:
    """SVM, обучаемый решением двойственной задачи по lambda.

    Решающее правило (обозначения лекции): a(x) = sign(sum_i lambda_i y_i K(x_i, x) - w0).
    """

    def __init__(self, C=1.0, kernel="linear", gamma=1.0, degree=3, coef0=1.0, eps=1e-6):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.eps = eps
        self._k = make_kernel(kernel, gamma, degree, coef0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        K = self._k(X, X)
        Q = (y[:, None] * y[None, :]) * K

        res = minimize(
            lambda lam: 0.5 * lam @ Q @ lam - lam.sum(),
            x0=np.zeros(n),
            jac=lambda lam: Q @ lam - 1.0,
            bounds=[(0.0, self.C)] * n,
            constraints=[{"type": "eq", "fun": lambda lam: lam @ y, "jac": lambda lam: y}],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        self.converged_ = bool(res.success)
        lam = np.clip(res.x, 0.0, self.C)
        lam[lam < self.eps * self.C] = 0.0
        lam[lam > self.C * (1 - self.eps)] = self.C

        self.X_, self.y_, self.lambdas_ = X, y, lam
        self.support_ = lam > 0
        self.boundary_ = self.support_ & (lam < self.C)
        self.violators_ = lam == self.C
        self.w0_ = self._compute_w0(K)
        self.dual_value_ = float(-res.fun)
        return self

    def _compute_w0(self, K):
        g = K @ (self.lambdas_ * self.y_)
        if self.boundary_.any():
            return float(np.mean(g[self.boundary_] - self.y_[self.boundary_]))
        # Граничных векторов нет (при малых C все опорные - нарушители): w0 неоднозначен.
        # Из условий KKT (y_i f_i >= 1 при lambda_i = 0, y_i f_i <= 1 при lambda_i = C)
        # получаем отрезок допустимых w0 и берём его середину.
        t = g - self.y_
        pos, zero = self.y_ > 0, self.lambdas_ == 0
        is_lower = (pos & ~zero) | (~pos & zero)
        lo = t[is_lower].max() if is_lower.any() else -np.inf
        hi = t[~is_lower].min() if (~is_lower).any() else np.inf
        return float((lo + hi) / 2) if np.isfinite(lo) and np.isfinite(hi) else float(lo if np.isfinite(lo) else hi)

    def decision_function(self, X):
        sv = self.support_
        K = self._k(np.asarray(X, dtype=float), self.X_[sv])
        return K @ (self.lambdas_[sv] * self.y_[sv]) - self.w0_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)

    def margins(self, X, y):
        return np.asarray(y, dtype=float) * self.decision_function(X)

    @property
    def w_(self):
        if self.kernel != "linear":
            raise ValueError("Явный вектор w определён только для линейного ядра")
        return (self.lambdas_ * self.y_) @ self.X_
