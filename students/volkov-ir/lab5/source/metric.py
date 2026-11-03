import numpy as np
import model

def calculate_accuracy(X, y, w):
    mult_vector = np.dot(X, w)
    coincide = np.sign(mult_vector) == np.sign(y)
    return np.mean(coincide)

def confusion_matrix(X, y, w):
    mult_vector = np.dot(X, w)
    predictions = np.sign(mult_vector)

    TP = (predictions == 1) & (y == 1)
    TN = (predictions == -1) & (y == -1)
    FP = (predictions == 1) & (y == -1)
    FN = (predictions == -1) & (y == 1)

    count_TP = np.sum(TP)
    count_TN = np.sum(TN)
    count_FP = np.sum(FP)
    count_FN = np.sum(FN)

    matrix = np.array([[count_TP, count_FN], [count_FP, count_TN]])

    return matrix, count_TP, count_TN, count_FP, count_FN

def calculate_precision(TP, FP):
    return TP/(TP+FP)

def calculate_recall(TP, FN):
    return TP/(TP+FN)

def calculate_f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))

def calculate_log_loss(X, y, w):
    margin = model.calculate_margin(X, w, y)
    loss = model.calculate_loss(margin)
    return model.calculate_Qmean(loss)
