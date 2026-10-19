import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes):
    """Rows are true classes, columns are predicted ones."""
    return np.array([[np.sum((y_true == t) & (y_pred == p)) for p in range(n_classes)] for t in range(n_classes)])


def class_metrics(cm):
    """Precision, recall and F1 of every class, computed from the confusion matrix."""
    correct = np.diag(cm)
    precision = np.divide(correct, cm.sum(axis=0), out=np.zeros(len(cm)), where=cm.sum(axis=0) > 0)
    recall = np.divide(correct, cm.sum(axis=1), out=np.zeros(len(cm)), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(cm)), where=precision + recall > 0)
    return precision, recall, f1
