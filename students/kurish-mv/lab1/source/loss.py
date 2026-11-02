import numpy as np

from margin import calculate_margin


def loss_from_margin(m):
    return (1.0 - m) ** 2


def mean_loss(X, y, w, b=0.0):
    return float(np.mean(loss_from_margin(calculate_margin(X, y, w, b))))


def objective(X, y, w, b=0.0, l2=0.0):
    # Q = среднее потерь + l2 * ||w||^2 (смещение не штрафуется)
    return mean_loss(X, y, w, b) + l2 * float(w @ w)
