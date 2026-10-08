import numpy as np

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
    # print(matrix)
    # matplotlib.pyplot.imshow(matrix)
    # plt.colorbar()
    # plt.show()
    return matrix, count_TP, count_TN, count_FP, count_FN

def calculate_precision(TP, FP):
    return TP/(TP+FP)

def calculate_recall(TP, FN):
    return TP/(TP+FN)

def calculate_f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))

def calculate_roc(X, y, w):
    mult_vector = np.dot(X, w)
    mult_vector_unic = np.unique(mult_vector)

    TPR_list = []
    FPR_list = []
    for threshold in mult_vector_unic:
        predictions = np.where(mult_vector >= threshold, 1, -1)
        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == -1))
        FN = np.sum((predictions == -1) & (y == 1))
        TN = np.sum((predictions == -1) & (y == -1))
        TPR_list.append(TP/(TP+FN))
        FPR_list.append(FP/(FP+TN))

    return TPR_list, FPR_list
