import numpy as np


def margins(w, X, y):
    return y * (X @ w)


def quadratic_loss(m):
    return (1 - m) ** 2


def loss_gradient(w, x, y):
    """Gradient of (1 - M)^2 with respect to w for one object, where M = y * <w, x>."""
    m = y * (x @ w)
    return -2 * (1 - m) * y * x


def empirical_risk(w, X, y):
    """Q(w) = mean loss over the whole sample."""
    return quadratic_loss(margins(w, X, y)).mean()


def sgd(X, y, w, lr=0.01, momentum=0.0, n_epochs=50, seed=0, forgetting=None):
    """SGD with momentum: v = gamma * v + (1 - gamma) * lr * grad, then w = w - v.

    Objects are reshuffled every epoch, momentum = 0 gives plain SGD.
    Q is estimated recurrently: Q = lam * loss(i) + (1 - lam) * Q, lam = 1 / len(y) by default.
    history keeps the recurrent estimate and the true risk over the whole sample after each epoch.
    """
    rng = np.random.default_rng(seed)
    lam = forgetting if forgetting is not None else 1 / len(y)
    v = np.zeros_like(w)
    Q = empirical_risk(w, X, y)
    history = {"Q": [Q], "risk": [Q]}
    for _ in range(n_epochs):
        for i in rng.permutation(len(y)):
            loss = quadratic_loss(y[i] * (X[i] @ w))
            v = momentum * v + (1 - momentum) * lr * loss_gradient(w, X[i], y[i])
            w = w - v
            Q = lam * loss + (1 - lam) * Q
        history["Q"].append(Q)
        history["risk"].append(empirical_risk(w, X, y))
    return w, history
