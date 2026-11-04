"""Собственная реализация PCA через сингулярное разложение.

Обозначения лекции «Линейные модели регрессии» (слайды 19–23):
F = V D Uᵀ, D = diag(√λ_1, …, √λ_n), где λ_j — собственные значения FᵀF;
столбцы U — главные направления, G = F U = V D — новые признаки, F̂ = G Uᵀ.
"""

import numpy as np


class PCA:
    """Метод главных компонент.

    Интерфейс совместим со sklearn: components_ хранит Uᵀ (строки — u_j),
    explained_variance_ — дисперсии компонент λ_j / (ℓ − 1).
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        n_objects, n_features = X.shape
        m = n_features if self.n_components is None else self.n_components
        if not 1 <= m <= min(n_objects, n_features):
            raise ValueError(f"n_components должно быть от 1 до {min(n_objects, n_features)}, получено {m}")

        # Лекция строит PCA для F без сдвига; центрирование делает F̂ = GUᵀ аппроксимацией
        # отклонений от среднего — так же поступает sklearn.
        self.mean_ = X.mean(axis=0)
        F = X - self.mean_
        V, d, Ut = np.linalg.svd(F, full_matrices=False)
        V, Ut = self._fix_signs(V, Ut)

        eigenvalues = d ** 2  # λ_j — собственные значения FᵀF
        self.n_components_ = m
        self.singular_values_ = d[:m]
        self.components_ = Ut[:m]
        self.eigenvalues_ = eigenvalues[:m]
        self.explained_variance_ = eigenvalues[:m] / (n_objects - 1)
        self.explained_variance_ratio_ = eigenvalues[:m] / eigenvalues.sum()
        # Полный спектр нужен для оценки эффективной размерности.
        self.all_eigenvalues_ = eigenvalues
        self._V = V[:, :m]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """G = F U."""
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """G = V D — то же, что F U, но без лишнего умножения."""
        self.fit(X)
        return self._V * self.singular_values_

    def inverse_transform(self, G: np.ndarray) -> np.ndarray:
        """F̂ = G Uᵀ."""
        return G @ self.components_ + self.mean_

    @staticmethod
    def _fix_signs(V: np.ndarray, Ut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SVD определено с точностью до знака пары (v_j, u_j).

        Фиксируем знак так, чтобы наибольшая по модулю координата u_j была
        положительной — тогда результат детерминирован.
        """
        idx = np.argmax(np.abs(Ut), axis=1)
        signs = np.sign(Ut[np.arange(Ut.shape[0]), idx])
        signs[signs == 0] = 1.0
        return V * signs, Ut * signs[:, None]
