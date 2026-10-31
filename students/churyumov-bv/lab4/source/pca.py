import numpy as np


class PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, F):
        self.mean = F.mean(axis=0)
        Fc = F - self.mean
        V, d, Ut = np.linalg.svd(Fc, full_matrices=False)       # F = V D U^T
        sign = np.sign(Ut[np.arange(len(d)), np.abs(Ut).argmax(axis=1)])
        Ut, V = Ut * sign[:, None], V * sign

        self.l, self.n = F.shape
        self.V, self.d = V, d
        self.eigvals = d ** 2                                   # lambda_j, по убыванию
        self.U = Ut.T                                           # n x n, все компоненты
        m = self.n if self.n_components is None else self.n_components
        self.components_ = Ut[:m]                               # m x n, строки - u_j^T
        self.explained_variance_ = self.eigvals[:m] / (self.l - 1)
        self.explained_variance_ratio_ = self.eigvals[:m] / self.eigvals.sum()
        return self

    def transform(self, F):
        return (F - self.mean) @ self.components_.T             # G = F U

    def inverse_transform(self, G):
        return G @ self.components_ + self.mean                 # F_hat = G U^T

    def fit_transform(self, F):
        return self.fit(F).transform(F)

    def error(self, m):
        return self.eigvals[m:].sum() / self.eigvals.sum()

    def effective_dim(self, eps):
        return next(m for m in range(1, self.n + 1) if self.error(m) <= eps)


class PCRegression:
    def __init__(self, m):
        self.m = m

    def fit(self, F, y):
        self.mean, self.y_mean = F.mean(axis=0), y.mean()
        V, d, Ut = np.linalg.svd(F - self.mean, full_matrices=False)
        m = self.m
        self.alpha = Ut[:m].T @ ((V[:, :m].T @ (y - self.y_mean)) / d[:m])   # alpha = sum_j u_j (v_j^T y) / sqrt(lambda_j)
        return self

    def predict(self, F):
        return (F - self.mean) @ self.alpha + self.y_mean
