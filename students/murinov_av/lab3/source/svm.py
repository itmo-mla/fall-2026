import numpy as np
from scipy.optimize import minimize


def linear_kernel(X1, X2):
    """Линейное ядро: K(x, x') = <x, x'>"""
    return np.dot(X1, X2.T)

def rbf_kernel(X1, X2, gamma=1.0):
    """Гауссовское ядро (RBF): K(x, x') = exp(-gamma * ||x - x'||^2)"""
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

class CustomSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lambdas = None
        self.b = 0
        self.support_vectors_ = None
        self.support_vectors_labels_ = None
        self.support_vectors_multipliers_ = None

    def _get_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self._get_kernel(X, X)
        
        H = np.outer(y, y) * K
        H += 1e-8 * np.eye(n_samples) 
        
        def objective(l):
            return 0.5 * np.dot(l, np.dot(H, l)) - np.sum(l)
            
        def gradient(l):
            return np.dot(H, l) - np.ones(n_samples)
            
        constraints = {'type': 'eq', 'fun': lambda l: np.dot(l, y)}
        bounds = [(0, self.C) for _ in range(n_samples)]
        l0 = np.zeros(n_samples)
        
        result = minimize(
            objective, l0, method='SLSQP', jac=gradient,
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        self.lambdas = np.clip(result.x, 0, self.C)
        
        threshold = 1e-5
        sv_idx = np.where(self.lambdas > threshold)[0]
        
        self.support_vectors_ = X[sv_idx]
        self.support_vectors_labels_ = y[sv_idx]
        self.support_vectors_multipliers_ = self.lambdas[sv_idx]
        
        margin_sv_idx = np.where((self.lambdas > threshold) & (self.lambdas < self.C - threshold))[0]
        
        if len(margin_sv_idx) > 0:
            b_values = []
            for i in margin_sv_idx:
                K_i = self._get_kernel(self.support_vectors_, X[i].reshape(1, -1)).flatten()
                b_i = np.sum(self.support_vectors_multipliers_ * self.support_vectors_labels_ * K_i) - y[i]
                b_values.append(b_i)
            self.b = np.mean(b_values)
        else:
            b_values = []
            for i in sv_idx:
                K_i = self._get_kernel(self.support_vectors_, X[i].reshape(1, -1)).flatten()
                b_i = np.sum(self.support_vectors_multipliers_ * self.support_vectors_labels_ * K_i) - y[i]
                b_values.append(b_i)
            self.b = np.mean(b_values)
            
    def decision_function(self, X):
        """Вычисляет значение sum(l_i * y_i * K(x_i, x)) - w0"""
        K = self._get_kernel(self.support_vectors_, X)
        return np.dot(self.support_vectors_multipliers_ * self.support_vectors_labels_, K) - self.b

    def predict(self, X):
        """Классификатор a(x) = sign(...)"""
        return np.sign(self.decision_function(X))