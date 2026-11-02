import numpy as np


class PCAFromSVD:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        # Центрируем признаки
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Сингулярное разложение: X = U S V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Собственные значения матрицы X^T X
        self.eigenvalues_ = S ** 2

        # Объяснённая дисперсия
        self.explained_variance_ = S ** 2 / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.eigenvalues_ / self.eigenvalues_.sum()
        )

        # Количество главных компонент
        max_components = len(S)
        k = max_components if self.n_components is None else self.n_components

        if not isinstance(k, (int, np.integer)) or not 1 <= k <= max_components:
            raise ValueError(f"n_components must be between 1 and {max_components}")

        # Первые k главных направлений
        self.components_ = Vt[:k]
        self.n_components_ = k

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_

        # Проекция на главные компоненты
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        # Восстановление исходных признаков
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=float)
        X_reconstructed = self.inverse_transform(self.transform(X))

        # Относительная квадратичная ошибка восстановления
        return (
            np.linalg.norm(X - X_reconstructed) ** 2
            / np.linalg.norm(X - self.mean_) ** 2
        )

    def effective_dimension(self, epsilon=0.05):
        # Минимальное k, при котором потеря дисперсии <= epsilon
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        return int(np.searchsorted(cumulative_variance, 1 - epsilon) + 1)
    