import numpy as np


def svd(mtx):
    u, s, v = np.linalg.svd(mtx, full_matrices=True)
    return u, s, v


class PCA:
    def __init__(self, x: np.ndarray):
        self.x = x
        self.v, self.s, self.u_t = svd(x)
        self.g = self.x @ self.u_t.T
        self.components = self.x.shape[1]

    def reconstruct(self):
        return self.g @ self.u_t[: self.components, :]

    def reduce_component(self, cnt=0):
        s_mtx = np.zeros((self.x.shape[0], self.x.shape[1] - cnt))
        np.fill_diagonal(s_mtx, self.s)
        self.g = self.v @ s_mtx
        self.components = self.x.shape[1] - cnt
