from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    quadratic_loss: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy_interval: tuple[float, float]


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Общее число наблюдений должно быть положительным")
    if not 0 <= successes <= total:
        raise ValueError("Некорректное число успешных наблюдений")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def classification_metrics(
    labels: ArrayLike,
    predictions: ArrayLike,
    scores: ArrayLike,
) -> ClassificationMetrics:
    y = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or predicted.shape != y.shape or score_values.shape != y.shape:
        raise ValueError("Метки, прогнозы и оценки должны иметь одинаковую форму")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Метки классов должны быть -1 или +1")
    if not np.all(np.isin(predicted, (-1.0, 1.0))):
        raise ValueError("Прогнозы должны быть -1 или +1")

    true_negative = int(np.count_nonzero((y == -1.0) & (predicted == -1.0)))
    false_positive = int(np.count_nonzero((y == -1.0) & (predicted == 1.0)))
    false_negative = int(np.count_nonzero((y == 1.0) & (predicted == -1.0)))
    true_positive = int(np.count_nonzero((y == 1.0) & (predicted == 1.0)))
    accuracy = (true_positive + true_negative) / len(y)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    positive_scores = score_values[y == 1.0]
    negative_scores = score_values[y == -1.0]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    roc_auc = float(
        (np.count_nonzero(comparisons > 0.0) + 0.5 * np.count_nonzero(comparisons == 0.0))
        / comparisons.size
    )
    quadratic_loss = 0.5 * float(np.mean((score_values - y) ** 2))

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        roc_auc=roc_auc,
        quadratic_loss=quadratic_loss,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        accuracy_interval=wilson_interval(true_positive + true_negative, len(y)),
    )


def roc_curve_points(
    labels: ArrayLike,
    scores: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    thresholds = np.r_[np.inf, np.sort(np.unique(score_values))[::-1], -np.inf]
    true_positive_rates = np.empty_like(thresholds)
    false_positive_rates = np.empty_like(thresholds)
    positives = np.count_nonzero(y == 1.0)
    negatives = np.count_nonzero(y == -1.0)
    for index, threshold in enumerate(thresholds):
        predicted_positive = score_values >= threshold
        true_positive_rates[index] = (
            np.count_nonzero(predicted_positive & (y == 1.0)) / positives
        )
        false_positive_rates[index] = (
            np.count_nonzero(predicted_positive & (y == -1.0)) / negatives
        )
    return false_positive_rates, true_positive_rates
