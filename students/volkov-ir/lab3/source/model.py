import numpy as np
from scipy.optimize import minimize


def calculate_gram(X):
    return X @ X.T

def calculate_kernel(X_1, X_2, kernel, d, gamma):
    G = X_1 @ X_2.T
    if kernel == "linear":
        return G
    if kernel == "poly":
        # gamma = 1 дает формулу лекции (<x, x'> + 1)^d
        return (gamma * G + 1) ** d
    if kernel == "rbf":
        # квадраты длин строк
        sq_1 = np.sum(X_1 ** 2, axis=1)
        sq_2 = np.sum(X_2 ** 2, axis=1)
        # квадраты расстояний всех пар
        D = sq_1[:, None] - 2 * G + sq_2[None, :]
        # из-за округления D бывает чуть меньше 0
        D = np.maximum(D, 0)
        return np.exp(-gamma * D)

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

def calculate_w0(lam, X, y, w, C, eps):
    # граничные объекты: eps < lam < C - eps
    mask = (lam > eps) & (lam < C - eps)
    # w0 по каждому граничному объекту
    w0_border = X[mask] @ w - y[mask]
    return np.mean(w0_border)

def calculate_w0_kernel(lam, K, y, C, eps):
    v = calculate_v(lam, y)
    mask = (lam > eps) & (lam < C - eps)
    w0_border = K[mask] @ v - y[mask]
    return np.mean(w0_border)

def calculate_predict(K, lam, y, w0):
    # K - ядро (новые объекты x train), y - метки train
    v = calculate_v(lam, y)
    score = K @ v - w0
    # знак: +1 злокачественная, -1 доброкачественная
    return np.where(score >= 0, 1.0, -1.0)

def calculate_fit_predict(X_fit, y_fit, X_eval, kernel, d, C, gamma):
    # обучаемся на X_fit, предсказываем для X_eval
    K_fit = calculate_kernel(X_fit, X_fit, kernel, d, gamma)
    lam, count = calculate_lambda(K_fit, y_fit, C)
    w0 = calculate_w0_kernel(lam, K_fit, y_fit, C, 1e-6)
    K_eval = calculate_kernel(X_eval, X_fit, kernel, d, gamma)
    return calculate_predict(K_eval, lam, y_fit, w0), lam

def calculate_grid_score(X, y, kernel, d, C, gamma, points, shape):
    # обучаемся на X и считаем score = K @ v - w0 в точках сетки для картинки
    K = calculate_kernel(X, X, kernel, d, gamma)
    lam, count = calculate_lambda(K, y, C)
    w0 = calculate_w0_kernel(lam, K, y, C, 1e-6)
    v = calculate_v(lam, y)
    score = calculate_kernel(points, X, kernel, d, gamma) @ v - w0
    # отступы объектов train, чтобы отметить явных нарушителей
    M = y * (K @ v - w0)
    return score.reshape(shape), lam, M
