import numpy as np
from model import calculate_margin, calculate_loss

def calculate_w_random(n):
    return np.random.uniform(-1/(2*n), 1/(2*n), size=n)


def calculate_w_correlation(X, y):
    a = X.T @ y
    b = (X*X).sum(axis=0)
    return a / b

def calculate_q_start(X, y, w, size):
    a = np.random.choice(len(X), size, replace=False)
    X_sub = X[a]
    y_sub = y[a]
    M = calculate_margin(X_sub, y_sub, w)
    Loss = calculate_loss(M)
    return np.mean(Loss)