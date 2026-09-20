def margins(w, X, y):
    return y * (X @ w)


def quadratic_loss(m):
    return (1 - m) ** 2


def loss_gradient(w, x, y):
    """Gradient of (1 - M)^2 with respect to w for one object, where M = y * <w, x>."""
    m = y * (x @ w)
    return -2 * (1 - m) * y * x
