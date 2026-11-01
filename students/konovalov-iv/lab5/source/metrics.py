import numpy as np


def classification_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": np.mean(y_pred == y_true), "precision": precision, "recall": recall, "f1": f1}


def confusion_matrix(y_true, y_pred):
    return np.array([
        [np.sum((y_true == -1) & (y_pred == -1)), np.sum((y_true == -1) & (y_pred == 1))],
        [np.sum((y_true == 1) & (y_pred == -1)), np.sum((y_true == 1) & (y_pred == 1))],
    ])
