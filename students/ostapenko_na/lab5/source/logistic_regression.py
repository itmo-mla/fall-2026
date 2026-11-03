"""Binary logistic regression fitted with Newton-Raphson or IRLS."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class LogisticRegressionCustom:
    """Minimize mean log-loss + 0.5 * alpha * ||w||^2 (bias unpenalized)."""

    def __init__(self, method='newton', alpha=1.0, max_iter=100, tol=1e-9):
        if method not in ('newton', 'irls'):
            raise ValueError('method must be newton or irls')
        self.method = method
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol

    def _objective(self, X, y, beta):
        z = X @ beta
        return np.logaddexp(0.0, z).mean() - np.mean(y * z) + 0.5 * self.alpha * np.dot(beta[1:], beta[1:])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError('invalid X/y shapes')
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must be binary (0, 1)')
        A = np.column_stack([np.ones(len(X)), X])
        n, p = A.shape
        reg = np.diag(np.r_[0., np.full(p - 1, self.alpha)])
        beta = np.zeros(p)
        self.loss_history_ = [self._objective(A, y, beta)]
        for _ in range(self.max_iter):
            z = A @ beta
            prob = expit(z)
            weights = np.maximum(prob * (1 - prob), 1e-12)
            hessian = (A.T @ (weights[:, None] * A)) / n + reg
            if self.method == 'newton':
                gradient = (A.T @ (prob - y)) / n + reg @ beta
                direction = np.linalg.solve(hessian, gradient)
                candidate = beta - direction
            else:
                # IRLS: weighted least-squares on working response, with ridge penalty.
                working_response = z + (y - prob) / weights
                rhs = A.T @ (weights * working_response) / n
                candidate = np.linalg.solve(hessian, rhs)
                direction = beta - candidate
            # Backtracking makes both solvers stable on difficult datasets.
            step = 1.0
            old_loss = self.loss_history_[-1]
            while step > 1e-10 and self._objective(A, y, beta - step * direction) > old_loss + 1e-12:
                step *= 0.5
            new_beta = beta - step * direction
            new_loss = self._objective(A, y, new_beta)
            self.loss_history_.append(new_loss)
            delta = np.linalg.norm(new_beta - beta)
            beta = new_beta
            if delta < self.tol * (1 + np.linalg.norm(beta)):
                break
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        self.n_iter_ = len(self.loss_history_) - 1
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
