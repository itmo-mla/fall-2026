import numpy as np


class CustomPCA:
    """
    Метод главных компонент (PCA), реализованный через сингулярное разложение (SVD).
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
    
    def fit(self, X):
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        _, D, Ut = np.linalg.svd(X_centered, full_matrices=False)
        
        self.singular_values_ = D
        self.explained_variance_ = (D ** 2) / (self.n_samples_ - 1)
        self.components_ = Ut
        
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)
        else:
            self.n_components_ = self.n_components
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_[:self.n_components_].T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, G):
        return G @ self.components_[:self.n_components_] + self.mean_
    
    def effective_dimension(self, epsilon=0.05):
        total_variance = np.sum(self.explained_variance_)
        cumulative_explained = np.cumsum(self.explained_variance_) / total_variance
        residual = 1.0 - cumulative_explained
        m = np.searchsorted(-residual, -(epsilon)) + 1
        return int(min(m, len(self.explained_variance_)))
    
    def reconstruction_error(self, X):
        total_variance = np.sum(self.explained_variance_)
        retained_variance = np.sum(self.explained_variance_[:self.n_components_])
        return (total_variance - retained_variance) / total_variance