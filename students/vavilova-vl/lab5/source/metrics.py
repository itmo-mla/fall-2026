"""Метрики качества бинарной классификации (лекция 6, слайды 3–10)."""

import numpy as np


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred))


def mean_log_loss(y01: np.ndarray, p: np.ndarray) -> float:
    """−(1/ℓ) Σ [y_i ln p_i + (1 − y_i) ln(1 − p_i)] — минус среднее логарифма правдоподобия."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p)))


def roc_auc(y01: np.ndarray, score: np.ndarray) -> float:
    """AUC — доля правильно упорядоченных пар (отрицательный, положительный); ничьи — за 1/2."""
    pos, neg = score[y01 == 1], score[y01 == 0]
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def confusion(y01: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.sum((pred == 1) & (y01 == 1))),
        "TN": int(np.sum((pred == 0) & (y01 == 0))),
        "FP": int(np.sum((pred == 1) & (y01 == 0))),
        "FN": int(np.sum((pred == 0) & (y01 == 1))),
    }


def roc_curve(y01: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Точки ROC-кривой: объекты сортируются по убыванию score, порог проходит между ними.

    Объекты с равным score добавляются одной точкой (как в алгоритме слайда 7).
    """
    order = np.argsort(-score, kind="stable")
    s, y = score[order], y01[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.diff(s) != 0, True]  # конец группы равных score
    tpr = np.r_[0.0, tps[last] / y.sum()]
    fpr = np.r_[0.0, fps[last] / (1 - y).sum()]
    return fpr, tpr
