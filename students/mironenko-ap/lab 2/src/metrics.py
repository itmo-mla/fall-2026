"""Метрики на numpy (без sklearn)."""
import numpy as np


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    # по классам: precision = TP / (TP+FP), recall = TP / (TP+FN)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1(y_true, y_pred, n_classes):
    _, _, f1 = precision_recall_f1(y_true, y_pred, n_classes)
    return float(f1.mean())