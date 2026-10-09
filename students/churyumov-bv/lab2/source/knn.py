# Задание: (полное задание в файле main.py)
# 2. реализовать алгоритм KNN с методом окна Парзена переменной ширины;
#    1. в качестве ядра можно использовать гауссово ядро;
# 3. подобрать параметр k методом скользящего контроля (LOO);
# 4. обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;

import numpy as np

# Функция вычисления расстояния Минковского
def rho(x, xi, p=2, w=None):
    x, xi = np.asarray(x), np.asarray(xi)
    if w is None:
        w = np.ones_like(x, dtype=float)
    return np.sum(w * np.abs(x - xi) ** p) ** (1 / p)


# 2.1. в качестве ядра можно использовать гауссово ядро;
def gaussian_kernel(r):
    return np.exp(-2 * r ** 2)

def distance_matrix(X, rho, Z=None):
    if Z is None:
        l = len(X)
        D = np.zeros((l, l))
        for i in range(l):
            for j in range(i + 1, l):
                D[i, j] = D[j, i] = rho(X[i], X[j])
        return D

    D = np.zeros((len(X), len(Z)))
    for i in range(len(X)):
        for j in range(len(Z)):
            D[i, j] = rho(X[i], Z[j])
    return D


def a(dist, y_train, Y, k, K):
    """
    a(x; X^l, k, K) = argmax_{y in Y} sum_i [y_i = y] * K( rho(x, x_i) / rho(x, x^(k+1)) )
    rho -> rho() (значения собраны в distance_matrix()),  K -> gaussian_kernel()

    dist     - расстояния rho(x, x_i) от x до всех объектов обучения (строка distance_matrix()), shape (l,)
    y_train  - ответы y_i, shape (l,)
    Y        - множество классов
    k        - число соседей (определяет ширину окна)
    K        - ядро K(r)
    """
    # h(x) = rho(x, x^(k+1)): расстояние до (k+1)-го соседа; rho -> rho(), dist -> distance_matrix()
    h = np.sort(dist)[k]
    h = max(h, 1e-12)

    # Gamma_y(x) = sum_i [y_i = y] * K(rho(x, x_i) / h); K -> gaussian_kernel()
    scores = {y: np.sum((y_train == y) * K(dist / h)) for y in Y}
    return max(scores, key=scores.get)      # argmax по y in Y


def loo(D, y, Y, k, K):
    r"""
    LOO(k, X^l) = sum_i [ a(x_i; X^l \ {x_i}, k) != y_i ]
    a -> a(),  rho(x_i, x_j) -> D = distance_matrix(X, rho)

    D  - матрица расстояний, D[i, j] = rho(x_i, x_j), shape (l, l)
    y  - ответы y_i, shape (l,)
    Y  - множество классов
    k  - число соседей
    K  - ядро K(r)
    """
    l = len(D)
    errors = 0
    for i in range(l):
        dist = np.delete(D[i], i)           # rho(x_i, x_j) для x_j из X \ {x_i}: строка D[i] из distance_matrix() без i-го элемента
        y_rest = np.delete(y, i)            # ответы на X \ {x_i}
        if a(dist, y_rest, Y, k, K) != y[i]:
            errors += 1                      # [a(...) != y_i]; a(...) -> a()
    return errors

def loo_curve(D, y, Y, ks, K):
    l = len(y)
    return [loo(D, y, Y, k, K) / l for k in ks]


def predict(D, y_train, Y, k, K):
    return np.array([a(D[i], y_train, Y, k, K) for i in range(len(D))])
