"""Проверка тождеств лекции 2 (запуск: python3 test_lecture.py, exit 0 = OK).

1. При k = 1 (длина контроля) CCV(Omega = X^L) = LOO-1NN / L, т.е.
   CCV(Omega) · L равно сумме LOO-ошибок 1NN.
2. Профильная формула CCV(Omega) = Σ_m Π(m, Omega)·C(L−1−m,l−1)/C(L−1,l)
   совпадает с прямым перебором всех C^L_l разбиений X^L = X^l ⊔ X^k
   для метода 1NN при k = 2 и k = 3.
"""

import sys
from itertools import combinations
from math import comb

sys.dont_write_bytecode = True

import numpy as np

from knn import rho_matrix
from prototype_selection import CCV, Pi


def loo_1nn_sum(X_L, y):
    """LOO-сумма 1NN: Σ_i [a(x_i; X^L \\ {x_i}) ≠ y_i] (прямой подсчёт)."""
    D = rho_matrix(np.asarray(X_L, float), np.asarray(X_L, float))
    y = np.asarray(y)
    errors = 0
    for i in range(len(X_L)):
        mask = np.ones(len(X_L), dtype=bool)
        mask[i] = False
        nn = np.flatnonzero(mask)[int(np.argmin(D[i][mask]))]
        errors += int(y[nn] != y[i])
    return errors


def brute_ccv(X_L, y, l, k):
    """Прямое определение CCV: 1/C^L_l Σ 1/k Σ_{x_i ∈ X^k} [a(x_i; X^l) ≠ y_i]."""
    X_L = np.asarray(X_L, float)
    y = np.asarray(y)
    L = len(X_L)
    D = rho_matrix(X_L, X_L)
    total = 0.0
    n_splits = 0
    for control in combinations(range(L), k):
        control = set(control)
        train = [j for j in range(L) if j not in control]
        err = 0
        for i in control:
            nn = train[int(np.argmin(D[i][train]))]
            err += int(y[nn] != y[i])
        total += err / k
        n_splits += 1
    assert n_splits == comb(L, l)
    return total / n_splits


def main():
    rng = np.random.RandomState(7)
    X_L = rng.randn(8, 2)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    L = len(X_L)
    Omega_full = set(range(L))

    # Тождество 1: k = 1 → CCV(Omega = X^L) · L = LOO-сумма 1NN = Π(1) · L.
    ccv1 = CCV(Omega_full, X_L, y, l=L - 1, k=1)
    loo = loo_1nn_sum(X_L, y)
    assert abs(ccv1 * L - loo) < 1e-12, (ccv1, loo)
    assert abs(Pi(1, X_L, y, Omega_full) * L - loo) < 1e-12
    print(f"k=1: CCV(Omega=X^L) * L = {ccv1 * L:.6f} == LOO-1NN = {loo}")

    # Тождество 2: профильная формула == прямой перебор, k = 2 и k = 3.
    for k in (2, 3):
        l = L - k
        formula = CCV(Omega_full, X_L, y, l=l, k=k)
        brute = brute_ccv(X_L, y, l=l, k=k)
        assert abs(formula - brute) < 1e-12, (k, formula, brute)
        print(f"k={k}: CCV по профилю = {formula:.6f} == прямой перебор = {brute:.6f}")

    # Тождество 3: малое k --- формулы слайда (k = 2, k = 3) через Π(m).
    l = L - 2
    p1 = Pi(1, X_L, y, Omega_full)
    p2 = Pi(2, X_L, y, Omega_full)
    assert abs(CCV(Omega_full, X_L, y, l=l, k=2)
                 - (p1 * l / (l + 1) + p2 / (l + 1))) < 1e-12
    l = L - 3
    p3 = Pi(3, X_L, y, Omega_full)
    assert abs(CCV(Omega_full, X_L, y, l=l, k=3)
                 - (p1 * l / (l + 2) + p2 * 2 * l / ((l + 1) * (l + 2))
                    + p3 * 2 / ((l + 1) * (l + 2)))) < 1e-12
    print("формулы слайда при k=2, k=3 сошлись с CCV(Omega)")

    print("OK: все тождества лекции выполнены")


if __name__ == "__main__":
    main()
