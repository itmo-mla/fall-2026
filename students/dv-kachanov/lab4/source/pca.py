import numpy as np


def center(F, mu=None):
    """Центрирование признаков: F0 = F - 1 mu^T.

    Если mu не задано, используется среднее по столбцам (по объектам).
    Возвращает (F0, mu), чтобы то же среднее можно было применить к контролю.
    """
    F = np.asarray(F, dtype=float)
    if mu is None:
        mu = F.mean(axis=0)
    mu = np.asarray(mu, dtype=float)
    return F - mu, mu


def standardize(F, mu=None, sigma=None):
    """Стандартизация признаков: F0 = (F - 1 mu^T) diag(sigma)^{-1}.

    Центрирование и масштабирование выполняются по обучающей выборке:
    mu_j — среднее, sigma_j — стандартное отклонение (ddof=0, как в
    sklearn.preprocessing.StandardScaler). Те же mu и sigma применяются
    к контролю, чтобы не было утечки. Возвращает (F0, mu, sigma).
    """
    F = np.asarray(F, dtype=float)
    if mu is None:
        mu = F.mean(axis=0)
    mu = np.asarray(mu, dtype=float)
    if sigma is None:
        sigma = F.std(axis=0, ddof=0)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma == 0):
        raise ValueError("Нельзя масштабировать признак с нулевым разбросом")
    F_centered, _ = center(F, mu)
    return F_centered / sigma, mu, sigma


def align_signs(U, U_ref):
    """Согласование знаков столбцов U с эталонными U_ref.

    SVD определяется с точностью до знака столбца, поэтому перед сравнением
    столбцы U умножаются на sign(u_j^T u_ref_j).
    """
    U = np.asarray(U, dtype=float)
    U_ref = np.asarray(U_ref, dtype=float)
    signs = np.sign(np.sum(U * U_ref, axis=0))
    signs[signs == 0] = 1.0
    return U * signs, signs


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
