import numpy as np


class PCA_SVD:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        n = X.shape[0]
        self.singular_values_ = S
        self.components_ = Vt
        self.explained_variance_ = S ** 2 / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        return self

    def transform(self, X, m=None):
        m = m or self.components_.shape[0]
        return (X - self.mean_) @ self.components_[:m].T

    def inverse_transform(self, Z, m=None):
        m = m or Z.shape[1]
        return Z @ self.components_[:m] + self.mean_
