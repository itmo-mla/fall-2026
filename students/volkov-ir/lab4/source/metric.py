import numpy as np


def calculate_rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))


def calculate_r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot
