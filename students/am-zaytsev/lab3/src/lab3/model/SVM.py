import numpy as np
from scipy.optimize import minimize

from .kernels import linear, polynomial, rbf


class SVM:
    def __init__(self, c, kernel=None):
        self.c = c
        self.w = None
        self.w0 = None
        self.lam = 0
        if kernel is None:
            kernel = rbf
        self.kernel = kernel

    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        K = self.kernel(x)  # (n, n) Gram matrix — computed once, outside fun
        H = np.outer(y, y) * K  # also λ-independent — once
        fun = lambda lam: -lam.sum() + 0.5 * lam @ H @ lam

        opt_res = minimize(
            fun=fun,
            x0=np.zeros(x.shape[0]),
            bounds=[(0, self.c)] * x.shape[0],
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda lam: lam @ y},
        )

        # assert opt_res.success
        self.lam = opt_res.x

        self.w = np.sum(self.lam[:, None] * y[:, None] * x, axis=0)
        sv = (self.lam > 1e-8) & (self.lam < self.c - 1e-8)
        self.w0 = np.mean(self.kernel(x[sv], x) @ (self.lam * y) - y[sv])

    def decision(self, x):
        return self.kernel(self.x, x).T @ (self.lam * self.y) - self.w0

    def predict(self, x):
        return (self.decision(x) > 0) * 2 - 1
