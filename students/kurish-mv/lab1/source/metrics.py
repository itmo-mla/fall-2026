import numpy as np
from scipy.stats import rankdata


def confusion_matrix_binary(y_true, y_pred):
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def _div(a, b):
    return float(a / b) if b else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    (_, fp), (_, tp) = confusion_matrix_binary(y_true, y_pred)
    return _div(tp, tp + fp)


def recall(y_true, y_pred):
    (_, _), (fn, tp) = confusion_matrix_binary(y_true, y_pred)
    return _div(tp, tp + fn)


def specificity(y_true, y_pred):
    (tn, fp), _ = confusion_matrix_binary(y_true, y_pred)
    return _div(tn, tn + fp)


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _div(2 * p * r, p + r)


def balanced_accuracy(y_true, y_pred):
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def roc_auc(y_true, scores):
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def find_threshold(y_true, scores):
    # порог по score, максимизирующий F1 (подбирается на validation)
    candidates = np.quantile(scores, np.linspace(0.05, 0.95, 91))
    f1s = [f1_score(y_true, np.where(scores >= t, 1, -1)) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def evaluate_classification(y_true, y_pred, scores=None):
    result = {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_acc": balanced_accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix_binary(y_true, y_pred),
    }
    if scores is not None:
        result["roc_auc"] = roc_auc(y_true, scores)
    return result
