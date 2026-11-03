from dataclasses import dataclass, field

import numpy as np


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # то же, что 1 / (1 + e^-z), но без переполнения exp при больших |z|


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])  # свободный член - ещё один признак f_0(x) = 1


def margins(F, y, w):
    return y * (F @ w)  # M_i = y_i w^T x_i, y_i in {-1, +1}

def log_loss(F, y, w):
    return np.logaddexp(0, -margins(F, y, w)).sum()


def gradient(F, y, w):
    sigma = sigmoid(margins(F, y, w))
    return -F.T @ ((1 - sigma) * y) # dQ/dw_j = -sum (1 - sigma_i) y_i f_j(x_i)


def hessian(F, y, w):
    sigma = sigmoid(margins(F, y, w))
    return F.T @ (((1 - sigma) * sigma)[:, None] * F)  # d2Q/dw_j dw_k = sum (1 - sigma_i) sigma_i f_j f_k = F^T D F


@dataclass
class History:
    w: list = field(default_factory=list)          # w^0, w^1, ...
    loss: list = field(default_factory=list)       # Q(w^t)
    grad_norm: list = field(default_factory=list)  # max_j |dQ/dw_j| в точке w^t
    dsigma: list = field(default_factory=list)     # max_i |sigma_i^t - sigma_i^{t-1}| - критерий остановки со слайда 9

    @property
    def n_iter(self):
        return len(self.dsigma)

    @property
    def w_star(self):
        return self.w[-1]


def _run(F, y, w0, direction, step, tol, max_iter):
    # общий цикл: w := w + h_t * direction(w); отличаются методы только способом вычисления direction
    hist = History()
    w = np.asarray(w0, dtype=float).copy()
    sigma = sigmoid(margins(F, y, w))

    def record(w):
        hist.w.append(w.copy())
        hist.loss.append(log_loss(F, y, w))
        hist.grad_norm.append(np.abs(gradient(F, y, w)).max())

    record(w)
    for _ in range(max_iter):
        w = w + step * direction(w)
        sigma_new = sigmoid(margins(F, y, w))
        record(w)
        hist.dsigma.append(np.abs(sigma_new - sigma).max())
        sigma = sigma_new
        if hist.dsigma[-1] < tol:
            break
    return hist


def newton_raphson(F, y, w0, step=1.0, tol=1e-12, max_iter=100):
    def direction(w):
        return -np.linalg.solve(hessian(F, y, w), gradient(F, y, w))  # систему решаем, гессиан не обращаем
    return _run(F, y, w0, direction, step, tol, max_iter)

def irls(F, y, w0, step=1.0, tol=1e-12, max_iter=100):
    def direction(w):
        M = margins(F, y, w)
        sigma = sigmoid(M)
        gamma = np.sqrt((1 - sigma) * sigma)             # веса объектов
        y_tilde = y * np.exp(-M / 2)                     # = y_i sqrt((1 - sigma_i) / sigma_i), без деления на sigma_i
        return np.linalg.lstsq(gamma[:, None] * F, y_tilde, rcond=None)[0]  # взвешенный МНК, F~^T F~ не строится
    return _run(F, y, w0, direction, step, tol, max_iter)


def irls_glm(F, y01, w0, step=1.0, tol=1e-12, max_iter=100):
    y = 2 * y01 - 1  # _run считает критерий остановки по отступам, поэтому ему нужны метки в {-1, +1}

    def direction(w):
        mu = sigmoid(F @ w) # c'(theta_i) = E y_i
        gamma = np.sqrt(np.maximum(mu * (1 - mu), 1e-300))  # sqrt(c''(theta_i) / phi_i); пол - на случай насыщения mu = 1
        y_tilde = (y01 - mu) / gamma                     # (y_i - c'(theta_i)) / (phi_i gamma_i)
        return np.linalg.lstsq(gamma[:, None] * F, y_tilde, rcond=None)[0]
    return _run(F, y, w0, direction, step, tol, max_iter)


def gradient_descent(F, y, w0, n_iter):
    L = np.linalg.eigvalsh(F.T @ F).max() / 4  # Q'' <= F^T F / 4, значит шаг 1/L гарантирует убывание Q
    w = np.asarray(w0, dtype=float).copy()
    loss = [log_loss(F, y, w)]
    for _ in range(n_iter):
        w = w - gradient(F, y, w) / L
        loss.append(log_loss(F, y, w))
    return w, np.array(loss)

def least_squares(F, y):
    return np.linalg.lstsq(F, y, rcond=None)[0]
