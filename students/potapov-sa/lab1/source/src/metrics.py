import numpy as np


def confusion_matrix(Y: np.ndarray, Y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp, tn, fp, fn = 0, 0, 0, 0
    for y, y_pred in zip(Y, Y_pred):
        if y == y_pred:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_pred == 1:
                fp += 1
            else:
                fn += 1
    return tp, tn, fp, fn
