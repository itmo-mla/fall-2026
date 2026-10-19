"""Отбор эталонов Omega ⊆ X^L (лекция 2).

    CCV(X^L) = 1/C^L_l Σ_{X^L = X^l ⊔ X^k} 1/k Σ_{x_i ∈ X^k} [a(x_i; X^l) ≠ y_i];
    при k = 1: CCV(X^L) = LOO(X^L) / L.

    Профиль компактности выборки X^L --- доля объектов x_i, у которых m-й
    сосед x^{(m)}_i лежит в другом классе::

        Π(m) = 1/L Σ_i [y_i ≠ y^{(m)}_i],  m = 1, ..., L − 1.

    Теорема (точное выражение CCV для метода 1NN)::

        CCV(X^L) = Σ_{m=1}^{k} Π(m) · C(L−1−m, l−1) / C(L−1, l).

    Профиль относительно Omega::

        Π(m, Omega) = 1/L Σ_i [y_i ≠ y^{(m|Omega)}_i],  m = 1, ..., |Omega|,

    где x^{(m|Omega)}_i --- m-й сосед объекта x_i среди множества Omega
    (сам объект x_i из соседей исключается). Тогда::

        CCV(Omega) = Σ_{m=1}^{k} Π(m, Omega) · C(L−1−m, l−1) / C(L−1, l).

    Жадная стратегия добавления эталонов::

        Omega := {по одному объекту каждого класса};
        повторять
            найти x ∈ X^L \\ Omega: CCV(Omega ∪ {x}) → min;
            Omega := Omega ∪ {x};
        пока CCV уменьшается.

Omega представлено множеством индексов объектов X^L.
"""

import numpy as np
from math import comb

from knn import rho_matrix


def Pi(m, X_L, y, Omega, D=None, w=None, p=2):
    """Профиль Π(m, Omega) = 1/L Σ_i [y_i ≠ y^{(m|Omega)}_i].

    Соседи ищутся среди Omega \\ {x_i}. Если m-го соседа нет
    (m > |Omega \\ {x_i}|), слагаемое объекта равно 0.
    """
    X_L = np.asarray(X_L, dtype=float)
    y = np.asarray(y)
    L = len(X_L)
    Omega = set(int(j) for j in Omega)
    if D is None:
        D = rho_matrix(X_L, X_L, w, p)
    errors = 0
    for i in range(L):
        candidates = sorted(j for j in Omega if j != i)
        if m > len(candidates) or len(candidates) == 0:
            continue
        order = np.argsort(D[i, candidates], kind="stable")
        errors += y[i] != y[candidates[order[m - 1]]]
    return errors / L


def CCV(Omega, X_L, y, l, k, D=None, w=None, p=2):
    """Полный скользящий контроль CCV(Omega) через профиль Π(m, Omega).

    CCV(Omega) = Σ_{m=1}^{k} Π(m, Omega) · C(L−1−m, l−1) / C(L−1, l),
    L = |X^L|, l --- длина обучения, k --- длина контроля (L = l + k).
    """
    X_L = np.asarray(X_L, dtype=float)
    L = len(X_L)
    assert L == l + k, "по лекции X^L = X^l ⊔ X^k, т.е. L = l + k"
    denom = comb(L - 1, l)
    value = 0.0
    for m in range(1, k + 1):
        if 0 <= l - 1 <= L - 1 - m:
            R = comb(L - 1 - m, l - 1) / denom
        else:
            R = 0.0
        value += Pi(m, X_L, y, Omega, D, w, p) * R
    return value


def initial_Omega(X_L, y):
    """Omega := {по одному объекту каждого класса}.

    Правило выбора представителя класса в лекции не фиксировано; здесь ---
    ближайший к центроиду своего класса объект (детерминированно).
    """
    X_L = np.asarray(X_L, dtype=float)
    y = np.asarray(y)
    Omega = set()
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        centroid = X_L[indices].mean(axis=0)
        chosen = indices[int(np.argmin(np.sum((X_L[indices] - centroid) ** 2,
                                               axis=1)))]
        Omega.add(int(chosen))
    return Omega


def predict_by_Omega(X_Omega, y_Omega, X, w=None, p=2):
    """Классификатор 1NN a(x; X^l ∩ Omega) на множестве эталонов Omega."""
    D = rho_matrix(np.asarray(X, dtype=float),
                   np.asarray(X_Omega, dtype=float), w, p)
    return np.asarray(y_Omega)[np.argmin(D, axis=1)]


def select_Omega(X_L, y, l, k, w=None, p=2):
    """Жадное добавление эталонов: x ∈ X^L \\ Omega с CCV(Omega ∪ {x}) → min,
    пока CCV уменьшается. Возвращает (Omega, CCV(Omega))."""
    X_L = np.asarray(X_L, dtype=float)
    y = np.asarray(y)
    D = rho_matrix(X_L, X_L, w, p)
    Omega = initial_Omega(X_L, y)
    current = CCV(Omega, X_L, y, l, k, D, w, p)
    while True:
        best_x = None
        best_value = current
        for x in range(len(X_L)):
            if x in Omega:
                continue
            value = CCV(Omega | {x}, X_L, y, l, k, D, w, p)
            if value < best_value:
                best_value = value
                best_x = x
        if best_x is None:
            break
        Omega.add(best_x)
        current = best_value
    return Omega, current
