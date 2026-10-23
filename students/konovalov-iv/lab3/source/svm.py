import numpy as np
from scipy.optimize import minimize


def linear_kernel():
    return lambda A, B: A @ B.T


def poly_kernel(degree=3):
    return lambda A, B: (A @ B.T + 1.0) ** degree


def rbf_kernel(gamma=1.0):
    def k(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0))
    return k


class KernelSVM:
    def __init__(self, kernel, C=1.0, tol=1e-6):
        self.kernel = kernel
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        n = len(y)
        K = self.kernel(X, X)
        Q = np.outer(y, y) * K

        res = minimize(
            lambda l: 0.5 * l @ Q @ l - l.sum(),
            np.zeros(n),
            jac=lambda l: Q @ l - 1.0,
            bounds=[(0.0, self.C)] * n,
            constraints=[{"type": "eq", "fun": lambda l: l @ y, "jac": lambda l: y}],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        lam = np.clip(res.x, 0.0, self.C)
        lam[lam < self.tol * self.C] = 0.0
        self.opt_result_ = res
        self.lambda_ = lam
        self.dual_value_ = -res.fun

        sv = lam > 0
        self.X_sv_, self.y_sv_, self.lambda_sv_ = X[sv], y[sv], lam[sv]
        self.sv_index_ = np.flatnonzero(sv)

        border = sv & (lam < self.C * (1 - 1e-4))
        if not border.any():
            border = sv
        self.w0_ = float(np.mean((lam * y) @ K[:, border] - y[border]))
        return self

    def decision_function(self, X):
        return (self.lambda_sv_ * self.y_sv_) @ self.kernel(self.X_sv_, X) - self.w0_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def sv_types(self):
        types = np.zeros(len(self.lambda_), dtype=int)
        types[self.lambda_ > 0] = 1
        types[self.lambda_ >= self.C * (1 - 1e-4)] = 2
        return types
