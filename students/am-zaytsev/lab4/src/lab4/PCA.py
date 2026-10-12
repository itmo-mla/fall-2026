import numpy as np


def svd(mtx):
    u, s, v = np.linalg.svd(mtx, full_matrices=True)
    return u, s, v


class PCA:
    def __init__(self, x: np.ndarray):
        self.x = x
        self.u, self.s, self.v = svd(x)

    def reconstruct(self):
        s_mtx = np.zeros((self.x.shape[0], self.x.shape[1]))
        np.fill_diagonal(s_mtx, self.s)

        return self.u @ s_mtx @ self.v
