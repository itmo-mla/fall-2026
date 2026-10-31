"""Метрики классификации"""
import numpy as np
from .core import log


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    return tp, tn, fp, fn


def prf(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f = 2 * p * r / (p + r + 1e-12)
    return p, r, f


def roc_auc(y_true, scores):
    """ROC-AUC """
    pos = scores[y_true == 1]
    neg = scores[y_true == -1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = np.r_[pos, neg]
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty_like(all_s)
    ranks[order] = np.arange(1, len(all_s) + 1)
    _, inv, counts = np.unique(all_s, return_inverse=True, return_counts=True)
    mean_r = np.array([ranks[inv == i].mean() for i in range(len(counts))])
    ranks = mean_r[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def report(y_true, y_pred, scores=None):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    p, r, f = prf(y_true, y_pred)
    out = dict(accuracy=accuracy(y_true, y_pred),
               precision=p, recall=r, f1=f,
               tp=tp, tn=tn, fp=fp, fn=fn)
    if scores is not None:
        out["roc_auc"] = roc_auc(y_true, scores)
    return out


def print_report(name, r):
    log(f"\n{name}")
    for k, v in r.items():
        if isinstance(v, float):
            log(f"  {k:10s}: {v:.4f}")
        else:
            log(f"  {k:10s}: {v}")