"""Логистическая регрессия в постановке слайдов 6-9 + 20 лекции 5.

Обозначения лекции:
    F       - матрица "объекты-признаки" (f_j(x_i)) размера l x n, БЕЗ bias-столбца;
    y_tilde - подписанные метки {-1, +1}, y_tilde_i = 2*y_i - 1 (слайд 20);
    M_i     - отступ M_i = y_tilde_i * w^T x_i (слайд 6);
    sigma_i - сигмоид ОТСТУПА sigma_i = sigma(M_i) = P(y_i | x_i) (слайды 7-8);
    Q(w)    - функционал Q(w) = sum_i L(M_i), L(M) = log(1 + exp(-M)) (слайд 6);
    D       - диагональная матрица весов D = diag((1 - sigma_i) * sigma_i) (слайд 8);
    F_tilde - взвешенная матрица F_tilde = diag(gamma) F,
              gamma_i = sqrt((1 - sigma_i) * sigma_i) (слайд 9);
    h_t     - величина шага, h_t = 1 (слайды 3, 7).

Градиент (слайд 7):  dQ/dw_j = -sum_i (1 - sigma_i) * y_tilde_i * f_j(x_i).
Гессиан (слайд 7):   d2Q/dw_j dw_k = sum_i (1 - sigma_i) * sigma_i * f_j(x_i) * f_k(x_i).
Ньютон (слайд 7):    w^{t+1} := w^t - h_t * (Q''(w^t))^{-1} Q'(w^t), h_t = 1.
IRLS (слайды 8-9):   рабочий ответ y_tilde_i / sigma_i, веса (1 - sigma_i) * sigma_i,
    взвешенная задача sum_i (1 - sigma_i) * sigma_i * (w^T x_i - y_tilde_i / sigma_i)^2 -> min,
    шаг w := w + h_t * (F_tilde^T F_tilde)^{-1} F_tilde^T y_work,
    где y_work_i = gamma_i * y_tilde_i / sigma_i = y_tilde_i * sqrt((1 - sigma_i) / sigma_i),
    что тождественно решению взвешенной задачи (домножение на gamma сводит
    взвешенный МНК к обычному МНК ||F_tilde * delta - y_work||^2).
Инициализация (слайд 9, п.1): w := (F^T F)^{-1} F^T y_tilde (обычный МНК).
Останов (слайд 9, п.9): выход, когда {sigma_i} мало изменились.

Численные (не модельные) приёмы: вырожденные системы решаются через lstsq/pinv,
деления на sigma_i защищены floor 1e-12. Регуляризации tau в лекции нет.
"""

import numpy as np

from autograd import Tensor

_H_T = 1.0
_SIGMA_FLOOR = 1e-12


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z, dtype=float)

    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result


def to_signed_labels(y01):
    """Перевод меток датасета {0, 1} в нотацию лекции {-1, +1} (слайд 20)."""
    return 2.0 * np.asarray(y01, dtype=float) - 1.0


def margins(F, y_tilde, w):
    """Отступы M_i = y_tilde_i * w^T x_i (слайд 6)."""
    return np.asarray(y_tilde, dtype=float) * (np.asarray(F, dtype=float) @ np.asarray(w, dtype=float))


def sigmas(F, y_tilde, w):
    """sigma_i = sigma(M_i) - сигмоид ОТСТУПА (слайд 7)."""
    return sigmoid(margins(F, y_tilde, w))


def objective_Q(F, y_tilde, w):
    """Q(w) = sum_i L(M_i), L(M) = log(1 + exp(-M)) (слайд 6)."""
    M = margins(F, y_tilde, w)
    return float(np.sum(np.logaddexp(0.0, -M)))


def autograd_gradient(F, y_tilde, w):
    """Градиент Q(w) через минимальный автоград (проверка замкнутой формулы слайда 7)."""
    F = np.asarray(F, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)
    w = np.asarray(w, dtype=float)

    F_tensor = Tensor(F)
    y_tensor = Tensor(y_tilde)
    w_tensor = Tensor(w)

    scores = F_tensor @ w_tensor
    M = scores * y_tensor
    loss_terms = (Tensor(1.0) + (-M).exp()).log()
    loss = loss_terms.mean() * float(F.shape[0])

    loss.backward()

    return w_tensor.grad.copy()


def gradient(F, y_tilde, w):
    """Q'(w): dQ/dw_j = -sum_i (1 - sigma_i) * y_tilde_i * f_j(x_i) (слайд 7)."""
    F = np.asarray(F, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)
    sigma = sigmas(F, y_tilde, w)
    return -F.T @ ((1.0 - sigma) * y_tilde)


def hessian(F, y_tilde, w):
    """Q''(w): d2Q/dw_j dw_k = sum_i (1 - sigma_i) * sigma_i * f_j * f_k (слайд 7)."""
    F = np.asarray(F, dtype=float)
    sigma = sigmas(F, y_tilde, w)
    weights = (1.0 - sigma) * sigma
    return (F.T * weights) @ F


def _solve_or_lstsq(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def least_squares_init(F, y_tilde):
    """Нулевое приближение w := (F^T F)^{-1} F^T y_tilde (слайд 9, п.1)."""
    F = np.asarray(F, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)
    return np.linalg.lstsq(F, y_tilde, rcond=None)[0]


def newton_raphson(F, y_tilde, h_t=_H_T, max_iter=100, tol=1e-10):
    """Метод Ньютона-Рафсона (слайд 7): w := w - h_t * (Q'')^{-1} Q', h_t = 1."""
    F = np.asarray(F, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)

    w = least_squares_init(F, y_tilde)
    sigma = sigmas(F, y_tilde, w)
    history = [objective_Q(F, y_tilde, w)]

    n_iter = 0
    for _ in range(max_iter):
        grad = gradient(F, y_tilde, w)
        H = hessian(F, y_tilde, w)
        step = _solve_or_lstsq(H, grad)
        w = w - h_t * step
        n_iter += 1
        history.append(objective_Q(F, y_tilde, w))

        sigma_new = sigmas(F, y_tilde, w)
        if float(np.max(np.abs(sigma_new - sigma))) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new

    return w, np.array(history), n_iter


def irls(F, y_tilde, h_t=_H_T, max_iter=100, tol=1e-10):
    """IRLS (слайды 8-9): w := w + h_t * (F_tilde^T F_tilde)^{-1} F_tilde^T y_work, h_t = 1."""
    F = np.asarray(F, dtype=float)
    y_tilde = np.asarray(y_tilde, dtype=float)

    w = least_squares_init(F, y_tilde)
    sigma = sigmas(F, y_tilde, w)
    history = [objective_Q(F, y_tilde, w)]

    n_iter = 0
    for _ in range(max_iter):
        sigma_safe = np.clip(sigma, _SIGMA_FLOOR, 1.0)
        gamma = np.sqrt((1.0 - sigma_safe) * sigma_safe)
        # D = diag((1 - sigma_i) * sigma_i) - веса слайда 8; F_tilde = diag(gamma) F.
        F_tilde = gamma[:, None] * F
        # Рабочий ответ слайда 8 y_tilde_i / sigma_i, пересчитанный к обычному
        # МНК ||F_tilde delta - y_work||^2: y_work_i = gamma_i * y_tilde_i / sigma_i.
        y_work = y_tilde * gamma / sigma_safe
        delta = np.linalg.lstsq(F_tilde, y_work, rcond=None)[0]
        w = w + h_t * delta
        n_iter += 1
        history.append(objective_Q(F, y_tilde, w))

        sigma_new = sigmas(F, y_tilde, w)
        if float(np.max(np.abs(sigma_new - sigma))) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new

    return w, np.array(history), n_iter


def decision(F, w):
    """Решающая функция лекции a(x, w) = sign(w^T x) (слайд 6), значения в {-1, +1}."""
    scores = np.asarray(F, dtype=float) @ np.asarray(w, dtype=float)
    return np.where(scores >= 0.0, 1.0, -1.0)


def predict01(F, w):
    """Предсказания в нотации датасета {0, 1} для подсчёта метрик."""
    return ((decision(F, w) + 1.0) / 2.0).astype(int)


def correct_class_proba(F, y_tilde, w):
    """sigma_i = P(y_i | x_i) - вероятности правильной классификации (слайд 8)."""
    return sigmas(F, y_tilde, w)
