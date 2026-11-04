import numpy as np


class PCA:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        V, d, Ut = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.V_ = V
        self.singular_values_ = d
        self.components_ = Ut
        self.n_objects_ = len(X)
        self.eigenvalues_ = d ** 2
        self.explained_variance_ = self.eigenvalues_ / (self.n_objects_ - 1)
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        return self

    def transform(self, X, m=None):
        return (X - self.mean_) @ self.components_[:m].T

    def inverse_transform(self, G):
        return G @ self.components_[:G.shape[1]] + self.mean_

    def residual_share(self):
        return np.concatenate(([1.0], 1 - np.cumsum(self.explained_variance_ratio_)))

    def effective_dimension(self, eps):
        return int(np.argmax(self.residual_share() <= eps))

    def steep_slope(self):
        E = self.residual_share()
        return E[:-2] / E[1:-1]
