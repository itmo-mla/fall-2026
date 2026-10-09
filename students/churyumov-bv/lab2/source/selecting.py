from math import comb
import numpy as np

def r_m(L, k):
    l = L - k
    return np.array([comb(L - 1 - m, l - 1) / comb(L - 1, l) for m in range(1, k + 1)])

def nearest_in(D, i, omega, k):
    idx = omega[omega != i]                              # Ω без самого x_i
    order = idx[np.argsort(D[i, idx])][:k + 1]           # x_i^(1|Ω), ..., x_i^(k+1|Ω); rho(x_i, x_j) -> D[i] из distance_matrix()
    return np.pad(order, (0, k + 1 - len(order)), constant_values=-1)


def ccv(D, y, omega, k):
    L = len(y)
    R = r_m(L, k)
    total = 0.0
    for i in range(L):
        idx = omega[omega != i]                      # Ω без самого x_i (как в nearest_in())
        order = idx[np.argsort(D[i, idx])][:k]       # k ближайших эталонов к x_i (как nearest_in(); rho -> D из distance_matrix())
        wrong = (y[order] != y[i])                   # [y_i != y_i^(m|Ω)], m = 1..k
        total += np.sum(wrong * R[:len(order)])      # sum_m [...] * R(m); R(m) -> r_m()
    return total / L                                 # (1/L) sum_i


def contributions(nn, y, R):
    first = nn[:, :len(R)]                           # k ближайших эталонов каждого x_i: списки nn из nearest_in()
    wrong = (y[first] != y[:, None]) & (first >= 0)  # [y_i != y_i^(m|Ω)]
    return wrong @ R                                 # T(x_i, Ω) = sum_m [...] * R(m); R(m) -> r_m(); mean(T) = CCV(Ω), как ccv()


def deletion_deltas(nn, y, R, T):
    k = len(R)
    delta = np.zeros(len(y))
    for p in range(k):
        removed = np.delete(nn, p, axis=1)           # список x_i без p-го соседа, запасной встаёт в строй
        T_new = contributions(removed, y, R)
        valid = nn[:, p] >= 0
        np.add.at(delta, nn[valid, p], (T_new - T)[valid])
    return delta


def select_prototypes(D, y, k_ctrl, tol=1e-9):
    L = len(y)
    R = r_m(L, k_ctrl)
    omega = np.arange(L)
    nn = np.array([nearest_in(D, i, omega, k_ctrl) for i in range(L)])
    T = contributions(nn, y, R)
    history = [T.mean()]
    best, stop = None, None                          # Ω и число удалённых объектов в момент остановки

    while len(omega) > k_ctrl + 1:
        delta = deletion_deltas(nn, y, R, T)
        x = omega[np.argmin(delta[omega])]
        if stop is None and (T.sum() + delta[x]) / L > history[-1] + tol:
            best, stop = omega, len(history) - 1     # CCV начнёт расти — фиксируем Ω, но удаляем дальше ради графика
        omega = omega[omega != x]
        for i in np.where((nn == x).any(axis=1))[0]:
            nn[i] = nearest_in(D, i, omega, k_ctrl)
        T = contributions(nn, y, R)
        history.append(T.mean())

    if stop is None:                                 # CCV не вырос до самого конца
        best, stop = omega, len(history) - 1
    return best, np.array(history), stop
