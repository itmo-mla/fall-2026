import numpy as np

EPS = 1e-5
PATIENCE = 2000
LAMBDA_Q = 0.01
Q0_SAMPLE_SIZE = 200


def margin(w, X, y):
    # X без bias-столбца, bias -- последний элемент w
    return y * (X @ w[:-1] + w[-1])


def loss(M):
    return (1.0 - M) ** 2


def loss_grad(w, x_i, y_i):
    # x_i уже с приписанной единицей под bias
    M_i = y_i * (x_i @ w)
    return -2.0 * (1.0 - M_i) * y_i * x_i


class FitHistory:
    def __init__(self):
        self.Q = []
        self.w_norm = []
        self.eta = []
        self.n_iter = 0
        self.stopped_early = False
        self.Q0 = np.nan


class LinearClassifier:
    def __init__(
        self,
        init="correlation",
        n_starts=1,
        step_strategy="fixed",
        eta=0.005,
        gamma=0.9,
        l2_lambda=0.0,
        sampling_strategy="uniform",
        margin_recompute_every=50,
        max_iter=20000,
        random_state=None,
    ):
        self.init = init
        self.n_starts = n_starts
        self.step_strategy = step_strategy
        self.eta = eta
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.sampling_strategy = sampling_strategy
        self.margin_recompute_every = margin_recompute_every
        self.max_iter = max_iter
        self.random_state = random_state

        self.w_ = None
        self.history_ = None
        self.multistart_histories_ = None

    def _init_weights(self, X, y, rng):
        d = X.shape[1]
        if self.init == "zeros":
            return np.zeros(d + 1)
        if self.init == "random":
            return rng.normal(0.0, 0.01, d + 1)

        # correlation: w_j ~ corr(x_j, y), bias с нуля
        std_x = X.std(axis=0)
        std_x[std_x == 0] = 1.0
        x_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        corr = (x_c.T @ y_c / X.shape[0]) / (std_x * y.std())
        return np.concatenate([corr, [0.0]])

    def margin(self, X, y, w=None):
        if w is None:
            w = self.w_
        return margin(w, X, y)

    def decision_function(self, X):
        w = self.w_
        return X @ w[:-1] + w[-1]

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(float)

    @staticmethod
    def steepest_eta(x_i, grad, l2_lambda, eps=1e-12):
        # eta* = ||g||^2 / (g^T H g), H -- гессиан (1-M)^2 + l2*||w||^2 вдоль grad,
        # при l2_lambda=0 сводится к 1/(2*||x_i||^2)
        xg = x_i @ grad
        den = 2.0 * (xg ** 2 + l2_lambda * (grad[:-1] @ grad[:-1]))
        return float(grad @ grad) / den if den > eps else 0.0

    def _fit_single_run(self, X, y, w0, rng):
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])

        w = w0.copy()
        v = np.zeros_like(w)
        reg_mask = np.ones_like(w)
        reg_mask[-1] = 0.0

        history = FitHistory()
        q0_idx = rng.choice(n, size=min(Q0_SAMPLE_SIZE, n), replace=False)
        M0 = y[q0_idx] * (X_aug[q0_idx] @ w)
        Q = float(np.mean(loss(M0)))
        history.Q0 = Q

        probs = None
        no_improve = 0
        it = 0
        for it in range(1, self.max_iter + 1):
            if self.sampling_strategy == "margin":
                if probs is None or it % self.margin_recompute_every == 0:
                    M_all = y * (X_aug @ w)
                    probs = 1.0 / (np.abs(M_all) + 1e-3)
                    probs /= probs.sum()
                i = rng.choice(n, p=probs)
            else:
                i = rng.integers(0, n)

            x_i = X_aug[i]
            y_i = y[i]

            grad = loss_grad(w, x_i, y_i) + 2.0 * self.l2_lambda * w * reg_mask
            eta_t = self.steepest_eta(x_i, grad, self.l2_lambda) if self.step_strategy == "steepest" else self.eta

            v = self.gamma * v + (1.0 - self.gamma) * grad
            w = w - eta_t * v

            M_i = y_i * (x_i @ w)
            L_i = float(loss(np.array([M_i]))[0])
            Q_new = LAMBDA_Q * L_i + (1.0 - LAMBDA_Q) * Q

            history.Q.append(Q_new)
            history.w_norm.append(float(np.linalg.norm(w)))
            history.eta.append(float(eta_t))

            if abs(Q_new - Q) < EPS:
                no_improve += 1
                if no_improve >= PATIENCE:
                    Q = Q_new
                    history.stopped_early = True
                    break
            else:
                no_improve = 0
            Q = Q_new

        history.n_iter = it
        return w, history

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        if self.n_starts <= 1:
            w0 = self._init_weights(X, y, rng)
            w, history = self._fit_single_run(X, y, w0, rng)
            self.w_ = w
            self.history_ = history
            self.multistart_histories_ = [history]
            return self

        best_w = None
        best_Q = np.inf
        histories = []
        for _ in range(self.n_starts):
            w0 = self._init_weights(X, y, rng)
            w, history = self._fit_single_run(X, y, w0, rng)
            histories.append(history)
            final_Q = history.Q[-1] if history.Q else history.Q0
            if final_Q < best_Q:
                best_Q = final_Q
                best_w = w

        self.w_ = best_w
        self.multistart_histories_ = histories
        self.history_ = min(histories, key=lambda h: h.Q[-1] if h.Q else h.Q0)
        return self
