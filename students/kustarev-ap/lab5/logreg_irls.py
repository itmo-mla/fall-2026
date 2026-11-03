import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def reg_vector(lam, d):
    v = np.full(d + 1, lam, dtype=float)
    v[0] = 0.0
    return v


def newton_step(X, y, w, lam):
    p = sigmoid(X @ w)
    wdiag = p * (1 - p)
    lam_vec = reg_vector(lam, X.shape[1] - 1)
    grad = X.T @ (y - p) - lam_vec * w
    H = -(X.T @ (X * wdiag[:, None]) + np.diag(lam_vec))
    return w - np.linalg.solve(H, grad)


def irls_step(X, y, w, lam):
    p = sigmoid(X @ w)
    wdiag = p * (1 - p)
    lam_vec = reg_vector(lam, X.shape[1] - 1)
    z = X @ w + (y - p) / wdiag
    A = X.T @ (X * wdiag[:, None]) + np.diag(lam_vec)
    b = X.T @ (wdiag * z)
    return np.linalg.solve(A, b)


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(X, y, lam, max_iter=50, tol=1e-10):
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1])
    history = [log_loss(y, sigmoid(Xb @ w))]
    for _ in range(max_iter):
        w = newton_step(Xb, y, w, lam)
        history.append(log_loss(y, sigmoid(Xb @ w)))
        if abs(history[-1] - history[-2]) < tol:
            break
    return w, history


def fit_with_norms(X, y, lam, max_iter=15):
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1])
    norms = [0.0]
    for _ in range(max_iter):
        try:
            w_new = newton_step(Xb, y, w, lam)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(w_new)):
            break
        w = w_new
        norms.append(np.linalg.norm(w))
    return norms
