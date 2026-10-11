import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, c):
        self.c = c
        self.w = None
        self.w0 = None
        self.lambda_list = 0

    def train(self, x: np.ndarray, y: np.ndarray):
        K = x @ x.T  # (n, n)
        H = np.outer(y, y) * K
        fun = lambda lam: -lam.sum() + 0.5 * lam @ H @ lam
        opt_res = minimize(
            fun=fun,
            x0=np.zeros(x.shape[0]),
            bounds=[(0, self.c)] * x.shape[0],
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda lambda_list: lambda_list @ y},
        )

        # assert opt_res.success
        self.lambda_list = opt_res.x

        self.w = np.sum(self.lambda_list[:, None] * y[:, None] * x, axis=0)
        sv = (self.lambda_list > 1e-8) & (self.lambda_list < self.c - 1e-8)
        self.w0 = (self.w @ x[sv].T - y[sv]).mean()

    def predict(self, x):
        assert self.w is not None
        assert self.w0 is not None
        return ((self.w @ x.T - self.w0) > 0) * 2 - 1
