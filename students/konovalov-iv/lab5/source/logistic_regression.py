import time

import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def margin(w, X, y):
    return (X @ w) * y


def log_loss(w, X, y):
    return np.sum(np.logaddexp(0, -margin(w, X, y)))


def gradient(w, X, y):
    sigma = sigmoid(margin(w, X, y))
    return -X.T @ ((1 - sigma) * y)


def hessian(w, X, y):
    sigma = sigmoid(margin(w, X, y))
    return X.T @ (X * ((1 - sigma) * sigma)[:, None])


def newton_direction(w, X, y):
    return np.linalg.solve(hessian(w, X, y), gradient(w, X, y))


def irls_direction(w, X, y):
    sigma = sigmoid(margin(w, X, y))
    gamma = (1 - sigma) * sigma
    X_weighted = X * np.sqrt(gamma)[:, None]
    y_weighted = y * np.sqrt((1 - sigma) / sigma)
    return np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]


class LogisticRegression:
    def __init__(self, method="newton", h=1.0, max_iter=100, tol=1e-10):
        self.method = method
        self.h = h
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.history = None

    def fit(self, X, y):
        if self.method == "irls":
            w = np.linalg.lstsq(X, y.astype(float), rcond=None)[0]
        else:
            w = np.zeros(X.shape[1])

        history = {"Q": [log_loss(w, X, y)], "grad_norm": [np.linalg.norm(gradient(w, X, y))]}
        sigma = sigmoid(margin(w, X, y))
        start = time.perf_counter()
        for _ in range(self.max_iter):
            if self.method == "newton":
                w = w - self.h * newton_direction(w, X, y)
            elif self.method == "irls":
                w = w + self.h * irls_direction(w, X, y)
            elif self.method == "gd":
                w = w - self.h * gradient(w, X, y)
            else:
                raise ValueError(f"unknown method: {self.method}")

            history["Q"].append(log_loss(w, X, y))
            history["grad_norm"].append(np.linalg.norm(gradient(w, X, y)))
            sigma_prev, sigma = sigma, sigmoid(margin(w, X, y))
            if np.max(np.abs(sigma - sigma_prev)) < self.tol:
                break

        history["time"] = time.perf_counter() - start
        self.w = w
        self.history = history
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w)

    def predict(self, X):
        return np.where(X @ self.w >= 0, 1, -1)
