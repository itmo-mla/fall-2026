import numpy as np


def _tp_fp_fn_tn(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    return tp, fp, fn, tn

def precision(y_true, y_pred):
    tp, fp, _, _ = _tp_fp_fn_tn(y_true, y_pred)
    return tp / (tp + fp + 1e-12)

def recall(y_true, y_pred):
    tp, _, fn, _ = _tp_fp_fn_tn(y_true, y_pred)
    return tp / (tp + fn + 1e-12)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)

def accuracy(y_true, y_pred):
    tp, fp, fn, tn = _tp_fp_fn_tn(y_true, y_pred)
    return (tp + tn) / (tp + fp + fn + tn + 1e-12)