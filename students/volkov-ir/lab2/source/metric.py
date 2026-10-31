import numpy as np


def calculate_accuracy(y_true, y_pred):
    coincide = y_pred == y_true
    return np.mean(coincide)


# по строкам настоящий вид, по столбцам предсказанный
def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def calculate_precision(TP, FP):
    # если вид ни разу не предсказан, точность считаем 0, как в sklearn
    if TP + FP == 0:
        return 0.0
    return TP/(TP+FP)


def calculate_recall(TP, FN):
    if TP + FN == 0:
        return 0.0
    return TP/(TP+FN)


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2*((precision*recall)/(precision+recall))


# F1 по каждому виду отдельно (вид против всех остальных), затем среднее
def calculate_macro_f1(y_true, y_pred, n_classes):
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    f1_list = []
    for c in range(n_classes):
        TP = matrix[c, c]
        FP = np.sum(matrix[:, c]) - TP
        FN = np.sum(matrix[c, :]) - TP
        f1_list.append(calculate_f1(calculate_precision(TP, FP), calculate_recall(TP, FN)))
    return np.mean(f1_list)
