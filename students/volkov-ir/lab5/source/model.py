import numpy as np


def calculate_z(X, w):
    return X @ w

def calculate_margin(X, w, y):
    mult_vector = np.dot(X, w)
    return np.multiply(mult_vector, y)

def calculate_sigmoid(per):
    return 1 / (1 + np.exp(-per))

def calculate_loss(margin):
    # если m >= 0, то np.log(1 + np.exp(-margin))
    # если m < 0,  то -margin np.log(1 + np.exp(margin))
    return np.maximum(0, -margin) + np.log(1 + np.exp(-np.abs(margin)))

def calculate_Q(loss):
    return np.sum(loss)

def calculate_Qmean(loss):
    return np.sum(loss)/len(loss)

def calculate_gradient(X, y, sigmoid_m):
    a = (1 - sigmoid_m) * y
    gradient = -X.T @ a
    return gradient

def calculate_hessian(X, sigmoid_m):
    d = (1 - sigmoid_m) * sigmoid_m
    h = X.T * d @ X
    return h

def init_w_zero(X):
    return np.zeros(X.shape[1])


def method_newton_raph(X, y, w, epsilon):
    count_q = 0
    limit = 100
    w_list = []
    Q_list = []
    Q_old = np.inf

    while True:

        margin = calculate_margin(X, w, y)
        loss = calculate_loss(margin)
        Q = calculate_Q(loss)
        count_q += 1

        Q_list.append(Q)
        w_list.append(w)

        if abs(Q - Q_old) <= epsilon:
            break
        Q_old = Q

        if count_q >= limit:
            break

        sigmoid_m = calculate_sigmoid(margin)
        gra = calculate_gradient(X, y, sigmoid_m)
        hes = calculate_hessian(X, sigmoid_m)
        delta_w = np.linalg.inv(hes) @ gra
        w = w - delta_w

    return w, Q, w_list, Q_list, count_q

def init_w(X, y):
    w  = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return w

def method_irls(X, y, w, epsilon):
    count_q = 0
    limit = 100
    w_list = []
    Q_list = []
    Q_old = np.inf

    while True:
        
        margin = calculate_margin(X, w, y)
        loss = calculate_loss(margin)
        Q = calculate_Q(loss)
        count_q += 1

        Q_list.append(Q)
        w_list.append(w)

        if abs(Q - Q_old) <= epsilon:
            break
        Q_old = Q

        if count_q >= limit:
            break

        sigmoid_m = calculate_sigmoid(margin)

        d = (1-sigmoid_m) * sigmoid_m
        gamma = np.sqrt(d)

        X_weighted = np.diag(gamma) @ X
        y_new = y * np.sqrt((1-sigmoid_m)/sigmoid_m)
        A = X_weighted.T @ X_weighted
        b = X_weighted.T @ y_new

        delta_w = np.linalg.inv(A) @ b
        w = w + delta_w

    return w, Q, w_list, Q_list, count_q