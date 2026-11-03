import numpy as np


def calculate_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def calculate_margins(K, lam, y, w0):
    # M = y * (сумма lam * y * K - w0)
    v = lam * y
    return y * (K @ v - w0)

def calculate_object_types(lam, C, eps):
    # периферийные lam = 0, граничные 0 < lam < C, нарушители lam = C
    periphery = lam <= eps
    border = (lam > eps) & (lam < C - eps)
    violators = lam >= C - eps
    return periphery, border, violators
