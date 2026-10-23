"""Логистическая регрессия: метод Ньютона-Рафсона, IRLS и градиентный спуск.

Метки y_i in {-1, +1}, F - матрица признаков со столбцом единиц (нулевой столбец - свободный член),
M_i = y_i <w, x_i> - отступ, sigma_i = sigma(M_i). Функционал с L2-штрафом (свободный член не штрафуется):

    Q(w) = sum_i ln(1 + exp(-M_i)) + tau/2 ||w_{1:}||^2,
    Q'(w)  = -F^T (y * (1 - sigma)) + tau D w,
    Q''(w) = F^T diag(sigma (1 - sigma)) F + tau D,    D = diag(0, 1, ..., 1).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

EPS = 1e-12


@dataclass
class History:
    w: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    step_norm: list = field(default_factory=list)

    @property
    def n_iter(self):
        return len(self.step_norm)

    @property
    def w_star(self):
        return self.w[-1]


def penalty_mask(n):
    d = np.ones(n)
    d[0] = 0.0
    return d


def margins(F, y, w):
    return y * (F @ w)


def log_loss(F, y, w, tau=0.0):
    return float(np.logaddexp(0.0, -margins(F, y, w)).sum() + 0.5 * tau * (penalty_mask(len(w)) * w) @ w)


def gradient(F, y, w, tau=0.0):
    sigma = expit(margins(F, y, w))
    return -F.T @ (y * (1.0 - sigma)) + tau * penalty_mask(len(w)) * w


def hessian(F, y, w, tau=0.0):
    sigma = expit(margins(F, y, w))
    return F.T @ ((sigma * (1.0 - sigma))[:, None] * F) + tau * np.diag(penalty_mask(len(w)))


def probability(F, w):
    """P(y = +1 | x) = sigma(<w, x>)."""
    return expit(F @ w)


def predict(F, w):
    return np.where(F @ w >= 0.0, 1, -1)


def newton_direction(F, y, w, tau=0.0):
    """-Q''^{-1} Q': систему решаем, гессиан не обращаем."""
    return -np.linalg.solve(hessian(F, y, w, tau), gradient(F, y, w, tau))


def irls_direction(F, y, w, tau=0.0):
    """Шаг как задача взвешенного МНК.

    gamma_i = sqrt(sigma_i (1 - sigma_i)) - вес объекта, y~_i = y_i sqrt((1 - sigma_i) / sigma_i) - модифицированный ответ.
    Тогда F~^T F~ = F^T diag(sigma(1-sigma)) F и F~^T y~ = F^T (y (1 - sigma)), то есть решается ровно ньютоновская система.
    Для штрафа к задаче дописываются строки sqrt(tau) D с ответом -sqrt(tau) D w. Матрица F~^T F~ не строится: используется lstsq.
    """
    sigma = np.clip(expit(margins(F, y, w)), EPS, 1.0 - EPS)
    gamma = np.sqrt(sigma * (1.0 - sigma))
    A = gamma[:, None] * F
    b = y * np.sqrt((1.0 - sigma) / sigma)
    if tau > 0:
        R = np.sqrt(tau) * np.diag(penalty_mask(len(w)))
        A, b = np.vstack([A, R]), np.concatenate([b, -R @ w])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def irls_glm_direction(F, y, w, tau=0.0):
    """Тот же шаг в записи GLM (метки 0/1): gamma_i = sqrt(c''(theta_i)) = sqrt(mu_i(1-mu_i)), y~_i = (y_i - mu_i)/gamma_i."""
    y01 = (y + 1) / 2
    mu = expit(F @ w)
    gamma = np.sqrt(np.maximum(mu * (1.0 - mu), 1e-300))
    A, b = gamma[:, None] * F, (y01 - mu) / gamma
    if tau > 0:
        R = np.sqrt(tau) * np.diag(penalty_mask(len(w)))
        A, b = np.vstack([A, R]), np.concatenate([b, -R @ w])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def fit(F, y, direction, tau=0.0, step=1.0, w0=None, tol=1e-10, max_iter=100):
    """w := w + h * direction(w). Останов по норме шага ||w^{t+1} - w^t|| < tol."""
    w = np.zeros(F.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    h = History()

    def record(w):
        h.w.append(w.copy())
        h.loss.append(log_loss(F, y, w, tau))
        h.grad_norm.append(float(np.linalg.norm(gradient(F, y, w, tau))))

    record(w)
    for _ in range(max_iter):
        move = step * direction(F, y, w, tau)
        h.step_norm.append(float(np.linalg.norm(move)))
        w = w + move
        record(w)
        if h.step_norm[-1] < tol:
            break
    return h


def newton_raphson(F, y, **kw):
    return fit(F, y, newton_direction, **kw)


def irls(F, y, **kw):
    return fit(F, y, irls_direction, **kw)


def irls_glm(F, y, **kw):
    return fit(F, y, irls_glm_direction, **kw)


def least_squares(F, y):
    """МНК на метках (нулевое приближение для IRLS и пример из вопроса 4)."""
    return np.linalg.lstsq(F, y, rcond=None)[0]


def gradient_descent(F, y, tau=0.0, n_iter=300, w0=None):
    """Градиентный спуск с шагом 1/L, L = lambda_max(F^T F)/4 + tau: Q'' <= F^T F / 4 + tau D."""
    w = np.zeros(F.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    L = np.linalg.eigvalsh(F.T @ F).max() / 4 + tau
    loss = [log_loss(F, y, w, tau)]
    for _ in range(n_iter):
        w = w - gradient(F, y, w, tau) / L
        loss.append(log_loss(F, y, w, tau))
    return w, np.array(loss)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def auc(y, score):
    """ROC AUC через статистику Манна-Уитни (ранги со средними для совпадений)."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
