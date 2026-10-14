import numpy as np


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)

        n_samples = X.shape[0]
        spectrum = (singular_values ** 2) / (n_samples - 1)
        total_variance = spectrum.sum()
        self.singular_values_ = singular_values[:self.n_components]
        self.explained_variance_ = spectrum[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_


def full_spectrum(X):
    X = np.asarray(X, dtype=float)

    X_centered = X - X.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = (singular_values ** 2) / (X.shape[0] - 1)

    return eigenvalues, eigenvalues / eigenvalues.sum()


def kaiser_dimension(eigenvalues):
    return int(np.sum(eigenvalues > eigenvalues.mean()))


def variance_threshold_dimension(ratios, threshold=0.95):
    return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)


def reconstruction_error(X, n_components):
    X = np.asarray(X, dtype=float)

    model = MyPCA(n_components=n_components).fit(X)
    restored = model.inverse_transform(model.transform(X))

    return float(np.mean((X - restored) ** 2))
