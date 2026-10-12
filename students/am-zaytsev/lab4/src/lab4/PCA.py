import numpy as np


def my_svd(X: np.ndarray):
    """
    Вычисляет SVD разложение матрицы X = U * S * Vt через собственные значения.
    Поддерживает прямоугольные матрицы (m x n).
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    k = min(m, n)

    # 1. Находим правые сингулярные векторы (V) и сингулярные числа через X.T @ X
    # Размер XT X: (n x n)
    XTX = X.T @ X
    eigvals_V, eigvectors_V = np.linalg.eigh(XTX)

    # Сортируем по убыванию (eigh возвращает по возрастанию)
    idx_V = np.argsort(eigvals_V)[::-1]
    eigvals_V = eigvals_V[idx_V]
    V = eigvectors_V[:, idx_V]

    s = np.sqrt(np.clip(eigvals_V[:k], 0, None))

    XXT = X @ X.T
    eigvals_U, eigvectors_U = np.linalg.eigh(XXT)

    idx_U = np.argsort(eigvals_U)[::-1]
    U = eigvectors_U[:, idx_U]

    U = U[:]
    Vt = V[:, :k].T

    X_proj = X @ Vt.T
    for i in range(k):
        if s[i] > 1e-9 and np.dot(U[:, i], X_proj[:, i]) < 0:
            U[:, i] = -U[:, i]

    return U, s, Vt


def svd(x):
    return my_svd(x)


class PCA:
    def __init__(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)

        self.x = x - self.mean
        self.v, self.s, self.u_t = svd(self.x)
        print(self.s)
        self.g = self.x @ self.u_t.T
        self.components = self.x.shape[1]

    def reconstruct(self):
        return self.g @ self.u_t[: self.components, :] + self.mean

    def reduce_component(self, cnt=0):
        s_mtx = np.zeros((self.x.shape[0], self.x.shape[1] - cnt))
        np.fill_diagonal(s_mtx, self.s)
        self.g = self.v @ s_mtx
        self.components = self.x.shape[1] - cnt

    @property
    def reduced(self):
        return self.g
