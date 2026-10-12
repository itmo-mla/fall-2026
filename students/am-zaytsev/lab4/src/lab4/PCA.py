import numpy as np


def svd(mtx):
    u, s, v = np.linalg.svd(mtx, full_matrices=True)
    return u, s, v


class PCA:
    def __init__(self, x: np.ndarray):
        self.x = x
        self.v, self.s, self.u_t = svd(x)
        self.g = self.x @ self.u_t.T

    def reconstruct(self):
        return self.g @ self.u_t
