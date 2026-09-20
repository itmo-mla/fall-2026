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


def sgd(X, y, w, lr=0.01, n_epochs=50, seed=0):
    """Plain SGD: one weight update per object, objects are reshuffled every epoch."""
    rng = np.random.default_rng(seed)
    history = [empirical_risk(w, X, y)]
    for _ in range(n_epochs):
        for i in rng.permutation(len(y)):
            w = w - lr * loss_gradient(w, X[i], y[i])
        history.append(empirical_risk(w, X, y))
    return w, history
