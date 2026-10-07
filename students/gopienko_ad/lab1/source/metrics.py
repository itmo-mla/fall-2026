import numpy as np

from students.gopienko_ad.lab1.source.linear_classifier import LinearClassifier, RidgeClassifier


def accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    return float(np.mean(y_true == y_pred))

def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eps: float = 1e-12
) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))

    return float(tp / (tp + fp + eps))

def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eps: float = 1e-12
) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    return float(tp / (tp + fn + eps))

def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eps: float = 1e-12
):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return float(2 * p * r / (p + r + eps))

def roc_auc(
        y_true: np.ndarray,
        scores: np.ndarray
) -> float:
    positive_mask = (y_true == 1)
    negative_mask = (y_true == -1)

    n_positive = int(np.sum(positive_mask))
    n_negative = int(np.sum(negative_mask))

    if (n_positive == 0 or n_negative == 0):
        return np.nan

    order = np.argsort(
        scores,
        kind="mergesort"
    )

    sorted_scores = scores[
        order
    ]

    ranks = np.zeros(
        len(scores),
        dtype=float
    )

    left = 0

    while left < len(scores):

        right = left + 1

        while (right < len(scores)
                and
                sorted_scores[right] == sorted_scores[left]
        ):
            right += 1

        average_rank = ((left + 1) + right) / 2

        ranks[order[left:right]] = average_rank
        left = right

    positive_rank_sum = np.sum(ranks[positive_mask])

    auc = ((positive_rank_sum - n_positive * (n_positive + 1) / 2)
           / (n_positive * n_negative))

    return float(auc)

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    tn = np.sum(
        (y_true == -1)
        & (y_pred == -1)
    )

    fp = np.sum(
        (y_true == -1)
        & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        & (y_pred == -1)
    )

    tp = np.sum(
        (y_true == 1)
        & (y_pred == 1)
    )

    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def calculate_metrics(
    model: LinearClassifier | RidgeClassifier,
    X: np.ndarray,
    y: np.ndarray
):
    prediction = model.predict(X)

    scores = model.decision_function(X)

    return {
        "accuracy": accuracy(
            y,
            prediction
        ),

        "precision": precision(
            y,
            prediction
        ),

        "recall": recall(
            y,
            prediction
        ),

        "f1": f1_score(
            y,
            prediction
        ),

        "roc_auc": roc_auc(
            y,
            scores
        ),
    }