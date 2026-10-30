import numpy as np

from gradient import calculate_gradient
from loss import loss_from_margin, objective
from margin import calculate_margin
from presentation import random_presentation


class SGDWithMomentum:
    def __init__(self, lr=0.01, momentum=0.9, l2=1e-3, epochs=60,
                 lr_decay=0.05, q_alpha=0.01,
                 presentation=random_presentation, seed=0):
        self.lr, self.momentum, self.l2 = lr, momentum, l2
        self.epochs, self.lr_decay, self.q_alpha = epochs, lr_decay, q_alpha
        self.presentation, self.seed = presentation, seed

    def fit(self, X, y, w0, b0=0.0):
        rng = np.random.default_rng(self.seed)
        w, b = w0.copy(), float(b0)
        vw, vb = np.zeros_like(w), 0.0
        q = objective(X, y, w, b, self.l2)
        history = {"recurrent_q": [], "full_q": [q]}

        for epoch in range(self.epochs):
            lr = self.lr / (1.0 + self.lr_decay * epoch)
            for i in self.presentation(X, y, w, b, rng):
                xi, yi = X[i:i + 1], y[i:i + 1]
                loss_i = loss_from_margin(calculate_margin(xi, yi, w, b))[0] + self.l2 * (w @ w)
                gw, gb = calculate_gradient(xi, yi, w, b, self.l2)
                vw = self.momentum * vw + gw
                vb = self.momentum * vb + gb
                new_w, new_b = w - lr * vw, b - lr * vb
                if not (np.isfinite(new_w).all() and np.isfinite(new_b)):
                    print("SGD разошёлся, остановка")
                    return w, b, history
                w, b = new_w, new_b
                q = self.q_alpha * loss_i + (1.0 - self.q_alpha) * q
                history["recurrent_q"].append(q)
            history["full_q"].append(objective(X, y, w, b, self.l2))
        return w, b, history


class SteepestGradientDescent:
    def __init__(self, l2=1e-3, max_iter=300, tol=1e-10):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, X, y, w0, b0=0.0):
        n, d = X.shape
        Xa = np.c_[X, np.ones(n)]
        R = np.eye(d + 1)
        R[-1, -1] = 0.0
        H = 2.0 / n * Xa.T @ Xa + 2.0 * self.l2 * R
        theta = np.append(w0, b0)
        history = {"recurrent_q": [], "full_q": [objective(X, y, w0, b0, self.l2)]}

        for _ in range(self.max_iter):
            gw, gb = calculate_gradient(X, y, theta[:-1], theta[-1], self.l2)
            g = np.append(gw, gb)
            if np.linalg.norm(g) < self.tol:
                break
            theta = theta - (g @ g) / (g @ H @ g) * g
            history["full_q"].append(objective(X, y, theta[:-1], theta[-1], self.l2))
        return theta[:-1], float(theta[-1]), history
