import numpy as np

from autograd import Tensor


def sigmoid(z):
    result = np.empty_like(np.asarray(z, dtype=float))

    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result


def probabilities(X, w):
    return sigmoid(X @ w)


def log_likelihood(X, y, w):
    p = probabilities(X, w)
    eps = 1e-12
    return float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))


def objective(X, y, w, l2=1.0):
    return -log_likelihood(X, y, w) + 0.5 * l2 * np.sum(w[1:] ** 2)


def autograd_gradient(X, y, w, l2=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    n_features = X.shape[1]

    X_tensor = Tensor(X)
    y_tensor = Tensor(y)
    w_tensor = Tensor(w)
    mask = Tensor(np.concatenate(([0.0], np.ones(n_features - 1))))

    scores = X_tensor @ w_tensor
    softplus = (Tensor(1.0) + scores.exp()).log()
    nll = (softplus + (-(y_tensor * scores))).mean() * float(len(X))
    reg = (w_tensor * mask * w_tensor).mean() * (0.5 * l2 * float(n_features))
    loss = nll + reg

    loss.backward()

    return w_tensor.grad.copy()


def gradient(X, y, w, l2=1.0):
    grad = X.T @ (probabilities(X, w) - y)
    grad[1:] = grad[1:] + l2 * w[1:]
    return grad


def hessian(X, w, l2=1.0):
    p = probabilities(X, w)
    weights = p * (1.0 - p)
    H = (X.T * weights) @ X
    H[1:, 1:] = H[1:, 1:] + l2 * np.eye(X.shape[1] - 1)
    return H


def newton_raphson(X, y, l2=1.0, max_iter=100, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w = np.zeros(X.shape[1])
    history = [objective(X, y, w, l2)]

    for _ in range(max_iter):
        step = np.linalg.solve(hessian(X, w, l2), autograd_gradient(X, y, w, l2))
        w = w - step
        history.append(objective(X, y, w, l2))

        if np.linalg.norm(step) < tol:
            break

    return w, np.array(history)


def irls(X, y, l2=1.0, max_iter=100, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w = np.zeros(X.shape[1])
    history = [objective(X, y, w, l2)]

    penalty = np.zeros(X.shape[1])
    penalty[1:] = l2

    for _ in range(max_iter):
        p = probabilities(X, w)
        weights = p * (1.0 - p)
        z = (X @ w) + (y - p) / np.maximum(weights, 1e-12)

        A = (X.T * weights) @ X + np.diag(penalty)
        b = (X.T * weights) @ z
        w = np.linalg.solve(A, b)
        history.append(objective(X, y, w, l2))

        if np.linalg.norm(autograd_gradient(X, y, w, l2)) < tol:
            break

    return w, np.array(history)


def predict_proba(X, w):
    return probabilities(np.asarray(X, dtype=float), w)


def predict(X, w, threshold=0.5):
    return (predict_proba(X, w) >= threshold).astype(int)
