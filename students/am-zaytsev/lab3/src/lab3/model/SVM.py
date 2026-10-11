from itertools import product
import numpy as np
from scipy.optimize import minimize, Bounds


class SVM:
    def __init__(self, c):
        self.c = c

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
            xo=np.zeros([x.shape[0], 1]),
            bounds=[Bounds(0, self.c)] * x.shape[0],
            constraints={"type": "eq", "fun": lambda lambda_list: lambda_list @ y},
        )
        