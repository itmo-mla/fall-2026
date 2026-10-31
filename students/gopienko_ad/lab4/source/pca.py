import numpy as np

class PCACustom:
	def __init__(self, n_components: int | None = None):
		self.n_components = n_components
		self.n_components_: np.ndarray | None = None
		self.mean_: np.ndarray | None = None
		self.components_: np.ndarray | None = None
		self.singular_values_: np.ndarray | None = None
		self.explained_variance_: np.ndarray | None = None
		self.explained_variance_ratio_: np.ndarray | None = None
		self.all_singular_values_: np.ndarray | None = None
		self.all_explained_variance_: np.ndarray | None = None
		self.all_explained_variance_ratio_: np.ndarray | None = None
		self.rank_: int | None = None

	def fit(self, X: np.ndarray) -> "PCACustom":
		X = np.asarray(X, dtype=float)
		self.mean_ = X.mean(axis=0)
		X_centered = X - self.mean_
		_, s, vt = np.linalg.svd(X_centered, full_matrices=False)

		eigenvalues = s ** 2 / (len(X) - 1)
		total_variance = float(eigenvalues.sum())

		explained_ratios = eigenvalues / total_variance

		max_components = len(s)
		if self.n_components is None:
			m = max_components
		else:
			m = self.n_components

		self.n_components_ = m
		self.rank_ = int(np.linalg.matrix_rank(X_centered))
		self.components_ = vt[:m]
		self.singular_values_ = s[:m]
		self.explained_variance_ = eigenvalues[:m]
		self.explained_variance_ratio_ = explained_ratios[:m]
		self.all_singular_values_ = s
		self.all_explained_variance_ = eigenvalues
		self.all_explained_variance_ratio_ = explained_ratios
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		return (X - self.mean_) @ self.components_.T

	def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
		Z = np.asarray(Z, dtype=float)
		return Z @ self.components_ + self.mean_

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
			return self.fit(X).transform(X)

	def effective_dimension(self, epsilon: float = 0.05) -> int:
		remaining = np.maximum(0.0, 1.0 - np.cumsum(self.all_explained_variance_ratio_))
		indices = np.flatnonzero(remaining <= epsilon + 1e-12)
		return int(indices[0] + 1) if indices.size else len(remaining)

	def reconstruction_error_ratio(self, n_components: int) -> float:
		return float(max(0.0, 1.0 - self.all_explained_variance_ratio_[:n_components].sum()))

	