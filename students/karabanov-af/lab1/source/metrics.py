import numpy as np


def confusion_matrix(y_true, y_pred, labels=(-1, 1)):
    """Rows are true classes, columns are predicted classes."""
    return np.array([[np.sum((y_true == t) & (y_pred == p)) for p in labels] for t in labels])


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, positive=1):
    predicted = y_pred == positive
    return float(np.sum(predicted & (y_true == positive)) / predicted.sum()) if predicted.any() else 0.0


def recall(y_true, y_pred, positive=1):
    actual = y_true == positive
    return float(np.sum(actual & (y_pred == positive)) / actual.sum()) if actual.any() else 0.0


def f1(y_true, y_pred, positive=1):
    p, r = precision(y_true, y_pred, positive), recall(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0
