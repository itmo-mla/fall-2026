import numpy as np


def _quadratic_loss(M):
    return 0.5 * (1.0 - M) ** 2


def _quadratic_loss_derivative(M):
    return M - 1.0


def _compute_gradient(X, y, w, w0, idx, reg_tau=0.0):
    x_i, y_i = X[idx], y[idx]

    M = (np.dot(x_i, w) + w0) * y_i
    dL_dM = _quadratic_loss_derivative(M)

    grad_w = dL_dM * y_i * x_i + reg_tau * w
    grad_w0 = dL_dM * y_i

    return grad_w, grad_w0, M


def _init_weights(X, y, init, rng):
    d = X.shape[1]

    if init == 'random':
        return rng.uniform(-1.0 / (2 * d), 1.0 / (2 * d), size=d)

    if init == 'correlation':
        w = np.dot(X.T, y) / (np.sum(X ** 2, axis=0) + 1e-12)
        return np.nan_to_num(w, nan=0.0)

    return np.zeros(d)


class QualityEstimator:
    """Рекуррентная оценка функционала качества."""

    def __init__(self, lam=0.01):
        self.lam = lam
        self.Q = None
        self.history = []

    def update(self, loss):
        loss = float(loss)

        if self.Q is None:
            self.Q = loss
        else:
            self.Q = self.lam * loss + (1.0 - self.lam) * self.Q

        self.history.append(self.Q)

        return self.Q


class BaseLinearClassifier:
    """Базовый класс для линейных классификаторов."""

    def decision_function(self, X):
        if getattr(self, 'w_', None) is None:
            raise RuntimeError('Модель не обучена')

        return np.dot(X, self.w_) + self.w0_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def margins(self, X, y, w=None, w0=None):
        if w is None:
            if getattr(self, 'w_', None) is None:
                raise RuntimeError('Модель не обучена')
            w = self.w_

        if w0 is None:
            if getattr(self, 'w0_', None) is None:
                raise RuntimeError('Модель не обучена')
            w0 = self.w0_

        return (np.dot(X, w) + w0) * y


class LinearSGDClassifier(BaseLinearClassifier):
    """
    Линейный классификатор с квадратичной функцией потерь.

    Обучение: стохастический градиентный спуск с инерцией.
    """

    def __init__(
        self,
        lr=0.01,
        gamma=0.9,
        epochs=100,
        lam_est=0.01,
        reg_tau=0.0,
        init='random',
        sampling='random',
        mu_plus=2.0,
        mu_minus=-1.0,
        seed=42
    ):
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs
        self.lam_est = lam_est
        self.reg_tau = reg_tau
        self.init = init
        self.sampling = sampling
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.seed = seed

        self.w_ = None
        self.w0_ = None
        self.q_history_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape

        w = _init_weights(X, y, self.init, rng).astype(np.float64)
        w0 = 0.0

        v_w = np.zeros(d)
        v_w0 = 0.0

        estimator = QualityEstimator(lam=self.lam_est)

        for epoch in range(self.epochs):
            if self.sampling == 'random':
                indices = rng.permutation(n)

            elif self.sampling == 'margin':
                margins = self.margins(X, y, w, w0)
                abs_m = np.abs(margins)

                mask = (margins <= self.mu_plus) & (margins >= self.mu_minus)

                prob = 1.0 / (abs_m + 1e-8)
                prob[~mask] = 0.0

                if prob.sum() == 0:
                    prob = np.ones(n)

                prob /= prob.sum()
                indices = rng.choice(n, size=n, p=prob)

            for idx in indices:
                grad_w, grad_w0, M = _compute_gradient(X, y,w,w0,idx, self.reg_tau)

                v_w = self.gamma * v_w + (1.0 - self.gamma) * grad_w
                v_w0 = self.gamma * v_w0 + (1.0 - self.gamma) * grad_w0

                w -= self.lr * v_w
                w0 -= self.lr * v_w0

                loss = _quadratic_loss(M) + 0.5 * self.reg_tau * np.sum(w ** 2)
                estimator.update(loss)

        self.w_ = w
        self.w0_ = w0
        self.q_history_ = estimator.history

        return self


class FastestGradientDescentClassifier(BaseLinearClassifier):
    """
    Линейный классификатор, обучаемый скорейшим градиентным спуском.
    """

    def __init__(
        self,
        epochs=100,
        reg_tau=0.0,
        lam_est=0.01,
        seed=42
    ):
        self.epochs = epochs
        self.reg_tau = reg_tau
        self.lam_est = lam_est
        self.seed = seed

        self.w_ = None
        self.w0_ = None
        self.q_history_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, _ = X.shape

        w = _init_weights(X, y, 'random', rng).astype(np.float64)
        w0 = 0.0

        estimator = QualityEstimator(lam=self.lam_est)

        for epoch in range(self.epochs):
            for idx in rng.permutation(n):
                grad_w, grad_w0, M = _compute_gradient(X, y, w, w0, idx, self.reg_tau)

                h_star = 1.0 / (np.dot(X[idx], X[idx]) + 1e-8)

                w -= h_star * grad_w
                w0 -= h_star * grad_w0

                loss = _quadratic_loss(M) + 0.5 * self.reg_tau * np.sum(w ** 2)
                estimator.update(loss)

        self.w_ = w
        self.w0_ = w0
        self.q_history_ = estimator.history

        return self