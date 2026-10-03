import numpy as np
from scipy.optimize import minimize


def linear(A, B):
    """Linear kernel K(x, x') = <x, x'> for every pair of rows: shape (len(A), len(B))."""
    return A @ B.T


def polynomial(degree=3, c=1.0):
    """Polynomial kernel K(x, x') = (<x, x'> + c)^degree: all monomials up to `degree`."""
    return lambda A, B: (A @ B.T + c) ** degree


def rbf(gamma=1.0):
    """Gaussian kernel K(x, x') = exp(-gamma ||x - x'||^2), ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>."""
    def kernel(A, B):
        sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * sq)
    return kernel


class SVM:
    """Soft margin SVM trained through the dual problem in lambda."""

    def __init__(self, C=1.0, kernel=linear, eps=1e-6):
        self.C = C
        self.kernel = kernel
        self.eps = eps

    def fit(self, X, y):
        """Solve  -sum(lam) + 1/2 lam^T Q lam -> min,  0 <= lam <= C,  sum(lam * y) = 0,
        where Q_ij = y_i y_j K(x_i, x_j).
        """
        Q = np.outer(y, y) * self.kernel(X, X)
        n = len(y)
        result = minimize(
            fun=lambda lam: 0.5 * lam @ Q @ lam - lam.sum(),
            jac=lambda lam: Q @ lam - 1,
            x0=np.zeros(n),
            bounds=[(0, self.C)] * n,
            constraints={"type": "eq", "fun": lambda lam: lam @ y, "jac": lambda lam: y.astype(float)},
            method="SLSQP",
            options={"maxiter": 1000},
        )
        lam = result.x

        # support vectors: only objects with lambda > 0 take part in the decision
        support = lam > self.eps
        self.lam, self.X_sv, self.y_sv = lam[support], X[support], y[support]

        # objects exactly on the margin (0 < lambda < C) satisfy  y_i (<w, x_i> - w0) = 1  =>  w0 = <w, x_i> - y_i
        on_margin = support & (lam < self.C - self.eps)
        self.w0 = np.median(self._score(X[on_margin]) - y[on_margin])
        return self

    @property
    def w(self):
        """w = sum_i lam_i y_i x_i, the normal of the separating hyperplane: only for the linear kernel."""
        return (self.lam * self.y_sv) @ self.X_sv

    def _score(self, X):
        """<w, x> = sum_i lam_i y_i K(x_i, x), w itself is never built."""
        return self.kernel(X, self.X_sv) @ (self.lam * self.y_sv)

    def decision_function(self, X):
        return self._score(X) - self.w0

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
