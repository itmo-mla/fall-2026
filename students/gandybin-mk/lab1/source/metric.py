from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred):
    """Возвращает accuracy, precision, recall и F1 для готовых предсказаний."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


def calculate_confusion_matrix(y_true, y_pred):
    """Вычисляет матрицу ошибок с фиксированным порядком классов [-1, +1]."""
    return confusion_matrix(y_true, y_pred, labels=[-1, 1])
