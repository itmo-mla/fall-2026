import time
import numpy as np
from scipy.optimize import minimize

from data_preprocessing import prepare_data


def calculate_gram(X):
    return X @ X.T

def calculate_v(lam, y):
    return lam * y

def calculate_dual_loss(lam, y, K):
    v = calculate_v(lam, y)
    # таблица vi * vj, l x l
    v_pairs = np.outer(v, v)
    return -np.sum(lam) + 0.5 * np.sum(v_pairs * K)

def calculate_dual_gradient(lam, y, K):
    v = calculate_v(lam, y)
    Kv = (K @ v)
    return y * Kv - 1

# ограничение вынесено внутрь calculate_lambda через lambda
# # сумма lam * y, в решении должна быть 0
# def calculate_constraint(lam, y):
#     return lam @ y
#
# def calculate_constraint_gradient(lam, y):
#     return y

def calculate_lambda(K, y, C):
    l = len(y)
    lam_start = np.zeros(l)
    bounds = [(0, C)] * l

    # сумма lam * y = 0
    constraint = {
        "type": "eq", # в решении функция должна быть равна 0
        "fun": lambda lam: lam @ y, # функция ограничителя
        "jac": lambda lam: y, # как считать градиент
    }

    result = minimize(calculate_dual_loss, lam_start, args=(y, K), method="SLSQP", jac=calculate_dual_gradient, bounds=bounds, constraints=constraint)

    # x - найденные лямбда, nit - число итераций
    return result.x, result.nit


def calculate_w(lam, X, y):
    v = calculate_v(lam, y)
    return X.T @ v


# решение двойственной задачи на train
X_train, X_test, y_train, y_test, feature_names = prepare_data()
K_train = calculate_gram(X_train)
C = 1


lam, count = calculate_lambda(K_train, y_train, C)

print("итераций:", count)
print("сумма lam * y:", lam @ y_train)
print("min lam:", np.min(lam), "max lam:", np.max(lam))
print("опорных:", np.sum(lam > 1e-6), "из", len(lam))
