import numpy as np


class PCA:
    def fit(self, features: np.ndarray) -> "PCA":
        if features.ndim != 2 or len(features) < 2:
            raise ValueError("Для PCA нужна матрица как минимум из двух объектов")

        self.feature_mean = features.mean(axis=0)
        centered_features = features - self.feature_mean
        _, self.singular_values, self.components = np.linalg.svd(
            centered_features, full_matrices=False
        )
        self.explained_variance = self.singular_values ** 2 / (len(features) - 1)
        total_variance = self.explained_variance.sum()
        if total_variance == 0:
            raise ValueError("Дисперсия данных равна нулю: выделить компоненты нельзя")
        self.explained_variance_ratio = self.explained_variance / total_variance
        return self

    def effective_dimension(self, variance_threshold: float = 0.95) -> int:
        if not 0 < variance_threshold <= 1:
            raise ValueError("Доля сохраняемой дисперсии должна лежать в (0, 1]")

        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        component_count = np.searchsorted(cumulative_variance, variance_threshold) + 1
        return min(int(component_count), len(self.components))

    def transform(
        self,
        features: np.ndarray,
        n_components: int | None = None,
    ) -> np.ndarray:
        if n_components is None:
            n_components = len(self.components)
        if not 1 <= n_components <= len(self.components):
            raise ValueError("Число компонент выходит за границы разложения")

        centered_features = features - self.feature_mean
        return np.dot(centered_features, self.components[:n_components].T)

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        n_components = coordinates.shape[1]
        if not 1 <= n_components <= len(self.components):
            raise ValueError("Число координат выходит за границы разложения")
        return np.dot(coordinates, self.components[:n_components]) + self.feature_mean
