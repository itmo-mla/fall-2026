from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

MODEL_LABELS = {
    "full": "Парзен, полная выборка",
    "sklearn": "sklearn KNN",
    "reduced": "Парзен, эталоны",
}


def quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Основные метрики классификации и матрица ошибок"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "errors": int(np.count_nonzero(y_true != y_pred)),
        "confusion": confusion_matrix(y_true, y_pred),
    }


def prediction_time_ms(model, X: np.ndarray, repeats: int = 51) -> float:
    """Медиана времени предсказания всей выборки X"""
    model.predict(X)
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict(X)
        timings.append((perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def prediction_changes(
    y_true: np.ndarray, y_before: np.ndarray, y_after: np.ndarray
) -> dict:
    """Сколько ответов исправлено и испорчено после замены модели"""
    return {
        "corrected": int(np.count_nonzero((y_before != y_true) & (y_after == y_true))),
        "introduced": int(np.count_nonzero((y_before == y_true) & (y_after != y_true))),
        "same": int(np.count_nonzero(y_before == y_after)),
    }
