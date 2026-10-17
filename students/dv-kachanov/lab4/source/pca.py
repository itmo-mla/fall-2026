"""Метод главных компонент в обозначениях лекции 4.

Матрица данных — F_{l x n} (l — число объектов, n — число признаков).
Центрирования и стандартизации НЕТ.

SVD лекции:  F = V D U^T,
  V_{l x n} — левые сингулярные векторы (с.в. матрицы F F^T),
  U_{n x n} — правые сингулярные векторы (с.в. матрицы F^T F),
  D = diag(sqrt(lambda_1), ..., sqrt(lambda_n)),
  lambda_j >= 0 — общие собственные значения F^T F и F F^T.

numpy.linalg.svd возвращает (U_svd, s, Vh) с F = U_svd diag(s) Vh,
поэтому отображение на лекционные обозначения такое:
  V = U_svd,  D = diag(s),  U = Vh^T,  lambda_j = s_j^2.
"""

import numpy as np


def scatter_matrix(F):
    """Рассеяние S = F^T F (без делителя)."""
    F = np.asarray(F, dtype=float)
    return F.T @ F


def spectrum(F):
    """Собственные значения lambda матрицы S = F^T F по убыванию."""
    S = scatter_matrix(F)
    return np.linalg.eigvalsh(S)[::-1]


def lecture_svd(F):
    """SVD в обозначениях лекции.

    Возвращает (V, d, U, lam):
      V_{l x n}, d — сингулярные числа (диагональ D), U_{n x n},
      lam — собственные значения S (lam_j = d_j^2), по убыванию.
    """
    F = np.asarray(F, dtype=float)
    U_svd, s, Vh = np.linalg.svd(F, full_matrices=False)
    V = U_svd
    U = Vh.T
    return V, s, U, s ** 2


def principal_components(F, U, m):
    """Новые признаки G_{l x m} = F U, где U_{n x m} — первые m столбцов U."""
    F = np.asarray(F, dtype=float)
    return F @ np.asarray(U, dtype=float)[:, :m]


def reconstruct(G, U, m):
    """Восстановление Fhat = G U^T по первым m столбцам U."""
    return np.asarray(G, dtype=float) @ np.asarray(U, dtype=float)[:, :m].T


def squared_error(F, Fhat):
    """Квадрат нормы Фробениуса ||Fhat - F||^2 (сумма, без усреднения)."""
    diff = np.asarray(Fhat, dtype=float) - np.asarray(F, dtype=float)
    return float(np.sum(diff ** 2))


def relative_errors(lam):
    """Относительные ошибки E_m = (lam_{m+1}+...+lam_n)/(lam_1+...+lam_n).

    Возвращает список E_0..E_n (E_0 = 1, E_n = 0).
    """
    lam = np.asarray(lam, dtype=float)
    total = float(np.sum(lam))
    return [float(np.sum(lam[m:]) / total) for m in range(len(lam) + 1)]


def effective_dimension(lam, eps):
    """Наименьшее m с E_m <= eps."""
    for m, e in enumerate(relative_errors(lam)):
        if e <= eps:
            return m
    return len(lam)


def ridge_alpha_star(F, y, tau):
    """Гребневая регрессия: alpha*_tau = (F^T F + tau I)^{-1} F^T y."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float)
    n = F.shape[1]
    return np.linalg.solve(scatter_matrix(F) + tau * np.eye(n), F.T @ y)


def ridge_functional(F, y, alpha, tau):
    """Функционал Q_tau(alpha) = ||F alpha - y||^2 + tau ||alpha||^2."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r = F @ alpha - y
    return float(r @ r + tau * (alpha @ alpha))


def control_functional(Fc, yc, alpha):
    """Функционал на контроле Q(alpha, X^k) = ||F' alpha - y'||^2."""
    Fc = np.asarray(Fc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r = Fc @ alpha - yc
    return float(r @ r)


def compressed_beta_star(G, y):
    """МНК на сжатых признаках (слайд 24): ||G beta - y||^2 -> min; beta = U^T alpha."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(G, y, rcond=None)
    return beta
