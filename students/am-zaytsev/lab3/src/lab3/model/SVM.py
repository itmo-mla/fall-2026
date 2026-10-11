from itertools import product

import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, c):
        self.c = c
        self.w = None
        self.w0 = None

    def train(self, x: np.array, y: np.array):

        def lagrangian(lambda_list):
            lagrangian_value = -np.sum(lambda_list)
            for i, j in product(
                range(x.shape[0]),
                range(x.shape[1]),
            ):
                lagrangian_value += (
                    lambda_list[i] * lambda_list[j] * y[i] * y[j] * (x[i] @ x[j].T)
                )
            return lagrangian_value

        opt_res = minimize(
            fun=lagrangian,
            x0=np.zeros(x.shape[0]),
            bounds=[(0, self.c)] * x.shape[0],
            constraints={"type": "eq", "fun": lambda lambda_list: lambda_list @ y},
        )

        assert opt_res.success
        self.lambda_list = opt_res.x

        self.w = np.sum(self.lambda_list[:, None] * y[:, None] * x, axis=0)
        self.w0 = (self.w @ x.T).mean()

    def predict(self, x):
        assert self.w is not None
        assert self.w0 is not None
        return ((self.w @ x.T + self.w0) > 0) * 2 - 1
