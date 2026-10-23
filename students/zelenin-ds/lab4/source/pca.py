import numpy as np


class PCASVD:
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCASVD":
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.singular_values_ = S
        self.components_ = Vt
        self.explained_variance_ = S**2 / (n - 1)
        total_variance = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA не обучена: сначала вызовите fit().")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        k = self.n_components
        Vt = self.components_ if k is None else self.components_[:k]
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ Vt.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        Vt = self.components_[: Z.shape[1]]
        return Z @ Vt + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
