import numpy as np


class PCASVD:
    """PCA через сингулярное разложение центрированной матрицы: X - mean = U diag(s) Vt.

    Строки Vt - главные оси (в лекции это столбцы U, а разложение записано как F = V D U^T).
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.n_samples_ = X.shape[0]
        self.mean_ = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.singular_values_ = s
        self.components_ = Vt
        self.explained_variance_ = s ** 2 / (self.n_samples_ - 1)
        self.explained_variance_ratio_ = s ** 2 / np.sum(s ** 2)
        return self

    def transform(self, X, k=None):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_[:k].T

    def inverse_transform(self, Z):
        return Z @ self.components_[:Z.shape[1]] + self.mean_

    def residual_share(self):
        """E_m = (lambda_{m+1} + ... + lambda_n) / (lambda_1 + ... + lambda_n), m = 0..n."""
        lam = self.singular_values_ ** 2
        return np.concatenate(([1.0], 1.0 - np.cumsum(lam) / lam.sum()))


def effective_dimension(E, eps=0.05):
    """Наименьшее m, при котором доля потерянной нормы E_m <= eps."""
    return int(np.argmax(np.asarray(E)[1:] <= eps) + 1)


def steep_slope(E, factor=3.0):
    """Первое m, на котором ошибка падает не менее чем в factor раз: E_{m-1} / E_m >= factor."""
    E = np.asarray(E)
    ratios = E[:-2] / np.maximum(E[1:-1], np.finfo(float).tiny)
    hit = np.flatnonzero(ratios >= factor)
    return int(hit[0] + 1) if hit.size else len(E) - 1


def kaiser(explained_variance):
    """Число компонент с дисперсией больше средней (для стандартизованных данных - больше 1)."""
    return int(np.sum(explained_variance > explained_variance.mean()))
