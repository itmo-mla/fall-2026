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
    order = np.argsort(dist)               # индексы объектов по возрастанию rho(x, x_i)

    # h(x) = rho(x, x^(k+1)): (k+1)-й сосед только задаёт ширину окна
    h = max(dist[order[k]], 1e-12)

    # голосуют только x^(1), ..., x^(k) — внутри окна, где r < 1
    nn = order[:k]

    # Gamma_y(x) = sum_{i <= k} [y_i = y] * K(rho(x, x_i) / h)
    scores = {y: np.sum((y_train[nn] == y) * K(dist[nn] / h)) for y in Y}
    return max(scores, key=scores.get)



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


def loo_prototypes(D, y, omega, Y, k, K):
    r"""
    LOO(k, Ω) = sum_{i=1..L} [ a(x_i; Ω \ {x_i}, k) != y_i ]
    Как CCV(Ω): контроль на всей выборке X^L, соседи — только из Ω.

    D      - матрица расстояний, D[i, j] = rho(x_i, x_j), shape (L, L)
    y      - ответы y_i, shape (L,)
    omega  - индексы эталонов Ω
    Y      - множество классов
    k      - число соседей
    K      - ядро K(r)
    """
    errors = 0
    for i in range(len(y)):
        idx = omega[omega != i]             # Ω без самого x_i (как в selecting.nearest_in())
        if a(D[i, idx], y[idx], Y, k, K) != y[i]:
            errors += 1                      # [a(...) != y_i]; a(...) -> a()
    return errors

def loo_prototypes_curve(D, y, omega, Y, ks, K):
    L = len(y)
    return [loo_prototypes(D, y, omega, Y, k, K) / L for k in ks]


def predict(D, y_train, Y, k, K):
    return np.array([a(D[i], y_train, Y, k, K) for i in range(len(D))])
