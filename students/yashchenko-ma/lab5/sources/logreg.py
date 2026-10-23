"""Логистическая регрессия: Ньютон-Рафсон и IRLS (лекция №5 "Нелинейная регрессия").

Обозначения лекции: y in {-1,+1}, M_i = y_i w^T x_i, sigma_i = sigma(M_i),
Q(w) = sum ln(1 + exp(-M_i))                                   (сл. 6)
Q'(w)_j  = -sum (1 - sigma_i) y_i f_j(x_i)                     (сл. 7)
Q''(w)_jk =  sum (1 - sigma_i) sigma_i f_j(x_i) f_k(x_i)       (сл. 7)
Свободный член вводится константным признаком f_0(x) = 1.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit


def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), np.asarray(X, float)])


def loss(w, F, y):
    """Q(w) = sum ln(1 + exp(-M_i)) (сл. 6)."""
    return float(np.sum(np.logaddexp(0.0, -y * (F @ w))))


def grad_hess(w, F, y):
    """Градиент и гессиан Q(w) (сл. 7). 1 - sigma_i = expit(-M_i) считается напрямую."""
    M = y * (F @ w)
    s, s_neg = expit(M), expit(-M)
    g = -F.T @ (s_neg * y)
    H = (F * (s_neg * s)[:, None]).T @ F          # F^T D F, D = diag((1-sigma) sigma)
    return g, H


@dataclass
class Result:
    w: np.ndarray
    w_hist: list = field(default_factory=list)
    loss_hist: list = field(default_factory=list)
    grad_hist: list = field(default_factory=list)   # ||Q'(w_t)||
    n_iter: int = 0
    converged: bool = False


def _init(F, y, init):
    if init == "ols":                               # сл. 9, шаг 1: обычный МНК
        return np.linalg.lstsq(F, y.astype(float), rcond=None)[0]
    return np.zeros(F.shape[1])


def _record(res, w, F, y):
    g, _ = grad_hess(w, F, y)
    res.w_hist.append(w.copy())
    res.loss_hist.append(loss(w, F, y))
    res.grad_hist.append(float(np.linalg.norm(g)))


def _run(step_fn, F, y, h, tol, max_iter, init):
    """Общий цикл; критерий остановки - как в лекции (сл. 9, шаг 9):
    max_i |sigma_i^{t+1} - sigma_i^t| < tol."""
    y = np.asarray(y, float)
    w = _init(F, y, init)
    res = Result(w=w)
    _record(res, w, F, y)
    for t in range(1, max_iter + 1):
        sigma_old = expit(y * (F @ w))
        w = w + h * step_fn(w, F, y)
        sigma_new = expit(y * (F @ w))
        _record(res, w, F, y)
        res.n_iter = t
        if np.max(np.abs(sigma_new - sigma_old)) < tol:
            res.converged = True
            break
    res.w = w
    return res


def _newton_step(w, F, y):
    """-(Q'')^{-1} Q' (сл. 7)."""
    g, H = grad_hess(w, F, y)
    return -np.linalg.solve(H, g)


def _irls_step(w, F, y):
    """Шаг IRLS (сл. 9): взвешенный МНК (F~^T F~)^{-1} F~^T y~."""
    M = y * (F @ w)
    s, s_neg = expit(M), expit(-M)
    gamma = np.sqrt(s_neg * s)                      # gamma_i = sqrt((1-sigma_i) sigma_i)
    Ft = gamma[:, None] * F                         # F~ = diag(gamma) F
    yt = y * np.sqrt(s_neg / s)                     # y~_i = y_i sqrt((1-sigma_i)/sigma_i)
    return np.linalg.lstsq(Ft, yt, rcond=None)[0]


def _irls_glm_step(w, F, y):
    """Шаг IRLS в форме GLM (сл. 18-21): Бернулли, y01 in {0,1}, theta = w^T x,
    c'(theta) = sigma(theta), c''(theta) = sigma (1 - sigma), phi = 1."""
    y01 = (y + 1.0) / 2.0
    mu = expit(F @ w)
    gamma = np.sqrt(mu * (1.0 - mu))                # sqrt(c''/phi)
    Ft = gamma[:, None] * F
    yt = (y01 - mu) / gamma                         # (y - c') / (phi * gamma)
    return np.linalg.lstsq(Ft, yt, rcond=None)[0]


STEPS = {"newton": _newton_step, "irls": _irls_step, "irls_glm": _irls_glm_step}


class LogisticRegressionIRLS:
    """method: 'newton' | 'irls' | 'irls_glm'; init: 'ols' (как в лекции) | 'zeros'."""

    def __init__(self, method="newton", h=1.0, tol=1e-10, max_iter=100, init="ols"):
        self.method, self.h, self.tol, self.max_iter, self.init = method, h, tol, max_iter, init

    def fit(self, X, y):
        F = add_intercept(X)
        self.res_ = _run(STEPS[self.method], F, y, self.h, self.tol, self.max_iter, self.init)
        self.intercept_, self.coef_ = self.res_.w[0], self.res_.w[1:]
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """P(y = +1 | x) = sigma(w^T x) (сл. 20-21)."""
        return expit(self.decision_function(X))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
