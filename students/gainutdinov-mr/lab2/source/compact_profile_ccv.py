import math

import numpy as np


def R(L, l, m):
    return math.comb(L - 1 - m, l - 1) / math.comb(L - 1, l)


def get_sorted_neighbours(X):
    # Для каждой точки x найти расстояние до каждого соседа
    # И затем получить список из соседей, отсортированный по расстоянию до каждого соседа
    # собрать из этого матрицу.
    X = np.asarray(X, dtype=float)

    # Матрица расстояний между всеми парами точек
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    # Исключаем саму точку из списка её соседей
    np.fill_diagonal(distances, np.inf)

    # Индексы соседей в порядке возрастания расстояния
    return np.argsort(distances, axis=1)[:, :-1]


def compact_profile(m, X, y):
    y = np.asarray(y)
    neighbours_m_indexes = get_sorted_neighbours(X)[
        :, m - 1
    ]  # TODO: Если выполнять в цикле, возможно стоит вынести в глобальный скоп
    labels_at_m = y[neighbours_m_indexes]
    # return np.sum(y != labels_at_m)/len(X)
    return np.mean(y != labels_at_m)


def compact_profile_M(M, X, y):
    y = np.asarray(y)
    neighbours_idx = get_sorted_neighbours(X)
    profile_at_m = []
    for m in range(1, M + 1):
        neighbours_m_indexes = neighbours_idx[:, m - 1]
        labels_at_m = y[neighbours_m_indexes]
        # return np.sum(y != labels_at_m)/len(X)
        profile = np.mean(y != labels_at_m)
        profile_at_m.append(profile)

    return profile_at_m


def compact_profile_M_v2(M, X, y):
    y = np.asarray(y)
    neighbours_idx = get_sorted_neighbours(X)[:, :M]
    neighbour_labels = y[neighbours_idx]

    mismatches = y[:, None] != neighbour_labels
    return np.mean(mismatches, axis=0)


def CompleteCrossValidation(X, y, l):
    """
    X - выборка
    y - правильные ответы
    l - количество обучающих объектов в обучающих выборках
    """
    L = len(X)
    denom_C = math.comb(L - 1, l)
    k = L - l
    summ = 0
    for m in range(1, k + 1):
        comp = compact_profile(m, X, y)
        coeff_C = math.comb(L - 1 - m, l - 1)
        summ += comp * coeff_C / denom_C

    return summ


def CompleteCrossValidation_v2(X, y, l):
    """
    X - выборка
    y - правильные ответы
    l - количество обучающих объектов в обучающих выборках
    """
    L = len(X)
    k = L - l
    comp = compact_profile_M(k, X, y)
    denom_C = math.comb(L - 1, l)
    coeffs_C = np.asarray(
        [math.comb(L - 1 - m, l - 1) / denom_C for m in range(1, k + 1)]
    )
    return np.sum(comp * coeffs_C)


def CompleteCrossValidation_v3(X, y, l):
    """
    X - выборка
    y - правильные ответы
    l - количество обучающих объектов в обучающих выборках
    """
    L = len(X)
    k = L - l
    comp = compact_profile_M_v2(k, X, y)
    denom_C = math.comb(L - 1, l)
    coeffs_C = np.asarray(
        [math.comb(L - 1 - m, l - 1) / denom_C for m in range(1, k + 1)]
    )
    return np.sum(comp * coeffs_C)
