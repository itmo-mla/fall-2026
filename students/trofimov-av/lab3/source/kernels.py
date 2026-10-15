import numpy as np


def kernel_matrix(x, z, kernel="linear", gamma=1.0):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    if kernel == "linear":
        return x @ z.T
    if kernel == "rbf":
        if gamma <= 0:
            raise ValueError("Параметр gamma должен быть положительным")
        distances = (
            np.sum(x ** 2, axis=1)[:, None]
            + np.sum(z ** 2, axis=1)[None, :]
            - 2 * x @ z.T
        )
        return np.exp(-gamma * np.maximum(distances, 0))
    raise ValueError("Ядро должно быть linear или rbf")
