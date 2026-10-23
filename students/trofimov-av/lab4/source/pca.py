import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
            raise ValueError("Ожидается конечная матрица минимум из двух объектов")
        if not isinstance(self.n_components, (int, np.integer)) or not (
            1 <= self.n_components <= min(x.shape)
        ):
            raise ValueError("Некорректное число компонент")
        self.mean_ = x.mean(axis=0)
        _, singular_values, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        variance = singular_values**2 / (len(x) - 1)
        total = variance.sum()
        self.components_ = vt[:self.n_components]
        self.singular_values_ = singular_values[:self.n_components]
        self.explained_variance_ = variance[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(self.n_components)
        )
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, z):
        return np.asarray(z, dtype=float) @ self.components_ + self.mean_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def effective_dimension(ratios, threshold=0.95):
    if not 0 < threshold <= 1:
        raise ValueError("Порог должен принадлежать (0, 1]")
    cumulative = np.cumsum(ratios)
    reached = np.flatnonzero(cumulative >= threshold - 1e-12)
    if len(reached) == 0:
        raise ValueError("Недостаточно компонент для выбранного порога")
    return int(reached[0] + 1)
