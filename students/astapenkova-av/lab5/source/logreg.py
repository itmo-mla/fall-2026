import numpy as np
from scipy.special import expit  # sigma(z) = 1 / (1 + exp(-z))


def sigmoid(z):
    return expit(z)


def log_loss(w, F, y, lam=0.0):
    """Q(w) = sum ln(1 + exp(-y_i w^T x_i)) + lam/2 ||w||^2."""
    margins = y * (F @ w)
    # ln(1 + e^{-M}) = logaddexp(0, -M)
    return np.sum(np.logaddexp(0.0, -margins)) + 0.5 * lam * (w @ w)


def gradient(w, F, y, lam=0.0):
    sigma = sigmoid(y * (F @ w))
    return -F.T @ ((1.0 - sigma) * y) + lam * w


def hessian(w, F, y, lam=0.0):
    sigma = sigmoid(y * (F @ w))
    d = (1.0 - sigma) * sigma
    return F.T @ (F * d[:, None]) + lam * np.eye(F.shape[1])


class NewtonRaphsonLogReg:
    def __init__(self, lam=1.0, step=1.0, max_iter=100, tol=1e-10):
        self.lam = lam
        self.step = step
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, F, y):
        w = np.zeros(F.shape[1])  
        self.history_ = [log_loss(w, F, y, self.lam)]
        self.grad_norms_ = [np.linalg.norm(gradient(w, F, y, self.lam))]

        for _ in range(self.max_iter):
            g = gradient(w, F, y, self.lam)
            H = hessian(w, F, y, self.lam)
            delta = np.linalg.solve(H, g)
            w = w - self.step * delta

            self.history_.append(log_loss(w, F, y, self.lam))
            self.grad_norms_.append(np.linalg.norm(gradient(w, F, y, self.lam)))
            if np.linalg.norm(delta) < self.tol:
                break

        self.coef_ = w
        self.n_iter_ = len(self.history_) - 1
        return self

    def predict_proba(self, F):
        return sigmoid(F @ self.coef_)

    def predict(self, F):
        return np.where(F @ self.coef_ >= 0, 1, -1)


class IRLSLogReg:
    # 1. w := (F^T F)^{-1} F^T y  
    # 2. для t = 1, 2, ...
    # 3.   sigma_i = sigma(y_i w^T x_i);
    # 4.   gamma_i = sqrt((1 - sigma_i) sigma_i);
    # 5.   F~ = diag(gamma) F;
    # 6.   y~_i = y_i sqrt((1 - sigma_i) / sigma_i);
    # 7.   выбрать шаг h_t;
    # 8.   w := w + h_t (F~^T F~)^{-1} F~^T y~;
    # 9.   если {sigma_i} мало изменились — выход.
    #     w := w + h_t (F~^T F~ + lam I)^{-1} (F~^T y~ - lam w)

    def __init__(self, lam=1.0, step=1.0, max_iter=100, tol=1e-10):
        self.lam = lam
        self.step = step
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, F, y):
        n = F.shape[1]
        I = np.eye(n)

        # шаг 1: обычный МНК 
        w = np.linalg.solve(F.T @ F + self.lam * I, F.T @ y)
        self.history_ = [log_loss(w, F, y, self.lam)]
        self.grad_norms_ = [np.linalg.norm(gradient(w, F, y, self.lam))]
        sigma_old = sigmoid(y * (F @ w))

        for _ in range(self.max_iter):
            # шаги 3–6
            sigma = np.clip(sigmoid(y * (F @ w)), 1e-15, 1 - 1e-15)
            gamma = np.sqrt((1.0 - sigma) * sigma)
            F_tilde = F * gamma[:, None]
            y_tilde = y * np.sqrt((1.0 - sigma) / sigma)

            # шаг 8: решение взвешенной задачи МНК ||F~ beta - y~||^2 (+ регуляризация)
            beta = np.linalg.solve(F_tilde.T @ F_tilde + self.lam * I,
                                   F_tilde.T @ y_tilde - self.lam * w)
            w = w + self.step * beta

            self.history_.append(log_loss(w, F, y, self.lam))
            self.grad_norms_.append(np.linalg.norm(gradient(w, F, y, self.lam)))

            # шаг 9: критерий остановки — веса sigma_i почти не меняются
            sigma_new = sigmoid(y * (F @ w))
            if np.max(np.abs(sigma_new - sigma_old)) < self.tol:
                break
            sigma_old = sigma_new

        self.coef_ = w
        self.n_iter_ = len(self.history_) - 1
        return self

    def predict_proba(self, F):
        return sigmoid(F @ self.coef_)

    def predict(self, F):
        return np.where(F @ self.coef_ >= 0, 1, -1)
