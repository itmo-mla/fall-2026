import numpy as np
from abc import ABC, abstractmethod


class LinearClassifier(ABC):
    def __init__(
        self,
        init="correlation",
        n_starts=5,
        epochs=100,
        momentum=0.9,
        learning_rate=0.1,
        forgetting=0.05,
        pool_size=200,
        batch_size=16,
        seed=42,
        sampling="combined", #margin/combined
    ):
        self.sampling = sampling
        self.init = init
        self.n_starts = n_starts
        self.epochs = epochs
        self.momentum = momentum
        self.lr = learning_rate
        self.forgetting = forgetting
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.seed = seed

        self.W = None
        self.b = None

        self.dW = None
        self.db = None

        self.vW = None
        self.vb = None

        self.Q_history = []
        self.step_history = []

    def _init_random(self, n_features, rng):
        self.W = rng.uniform(-1 / (2 * n_features), 1 / (2 * n_features), size=n_features)
        self.b = 0.0

    def _init_correlation(self, X, y):
        self.W = (X.T @ y) / np.sum(X ** 2, axis=0)
        self.b = float(np.mean(y))

    def _feedforward(self, X):
        return X @ self.W + self.b

    def predict_scores(self, X):
        return self._feedforward(X)

    def predict(self, X):
        return np.where(self.predict_scores(X) >= 0, 1, -1)

    def margins(self, X, y):
        return y * self._feedforward(X)

    def _compute_emp_risk(self, X, y):
        margins = self.margins(X, y)
        return np.mean((1 - margins) ** 2)

    def _compute_gradient(self, X_batch, y_batch):
        margins = self.margins(X_batch, y_batch)
        common = 2 * (margins - 1) * y_batch

        self.dW = (X_batch.T @ common) / len(y_batch)
        self.db = float(np.mean(common))

    def _select_batch(self, X, y, rng):
        n_batch = min(self.batch_size, len(X))
        if self.sampling == "random":
            indices = rng.choice(len(X), size=n_batch, replace=False)
            return X[indices], y[indices]

        absolute_margins = np.abs(self.margins(X, y))
        if self.sampling == "margin":
            indices = np.argsort(absolute_margins, kind="stable")[:n_batch]
            return X[indices], y[indices]

        n_pool = min(self.pool_size, len(X))
        pool = np.argpartition(absolute_margins, n_pool - 1)[:n_pool]

        n_batch = min(self.batch_size, n_pool)
        indices = rng.choice(pool, size=n_batch, replace=False)

        return X[indices], y[indices]

    @abstractmethod
    def _update(self, X_batch):
        raise NotImplementedError

    def _fit_once(self, X, y, rng):
        self.vW = np.zeros_like(self.W)
        self.vb = 0.0

        self.Q_history = []
        self.step_history = []

        Q = None

        for _ in range(self.epochs):
            X_batch, y_batch = self._select_batch(X, y, rng)

            batch_risk = self._compute_emp_risk(X_batch, y_batch)
            self._compute_gradient(X_batch, y_batch)
            step = self._update(X_batch)

            if Q is None:
                Q = batch_risk
            else:
                Q = self.forgetting * batch_risk + (1 - self.forgetting) * Q

            self.Q_history.append(Q)
            self.step_history.append(step)

    def fit(self, X, y, X_val=None, y_val=None):
        if self.init == "correlation":
            rng = np.random.default_rng(self.seed)
            self._init_correlation(X, y)
            self._fit_once(X, y, rng)

        else:  # multistart
            best_val_risk = np.inf
            best_state = None

            for start in range(self.n_starts):
                rng = np.random.default_rng(self.seed + start)

                self._init_random(X.shape[1], rng)
                self._fit_once(X, y, rng)

                val_risk = self._compute_emp_risk(X_val, y_val)

                if val_risk < best_val_risk:
                    best_val_risk = val_risk
                    best_state = (
                        self.W.copy(),
                        float(self.b),
                        self.Q_history.copy(),
                        self.step_history.copy()
                    )

            (self.W, self.b, self.Q_history, self.step_history) = best_state


class L2LinearClassifier(LinearClassifier):
    def __init__(self, *args, l2=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2 = l2

    def _compute_emp_risk(self, X, y):
        return super()._compute_emp_risk(X, y) + self.l2 / 2 * np.sum(self.W ** 2)

    def _compute_gradient(self, X_batch, y_batch):
        super()._compute_gradient(X_batch, y_batch)
        self.dW += self.l2 * self.W


class LinearClassifierMomentum(L2LinearClassifier):
    def _update(self, X_batch):
        self.vW = self.momentum * self.vW + (1 - self.momentum) * self.dW
        self.vb = self.momentum * self.vb + (1 - self.momentum) * self.db

        self.W -= self.lr * self.vW
        self.b -= self.lr * self.vb

        return self.lr


class LinearClassifierSpeediest(LinearClassifier):
    def _update(self, X_batch):
        squared_gradient = np.sum(self.dW ** 2) + self.db ** 2

        score_change = X_batch @ self.dW + self.db
        curvature = 2 * np.mean(score_change ** 2)

        step = squared_gradient / curvature if curvature > 0 else 0.0

        self.W -= step * self.dW
        self.b -= step * self.db

        return step


class LinearClassifier_mixed_version(L2LinearClassifier):
    def _update(self, X_batch):
        self.vW = self.momentum * self.vW + (1 - self.momentum) * self.dW
        self.vb = self.momentum * self.vb + (1 - self.momentum) * self.db

        gradient_along_momentum = self.dW @ self.vW + self.db * self.vb
        score_change = X_batch @ self.vW + self.vb
        curvature = 2 * np.mean(score_change ** 2) + self.l2 * np.sum(self.vW ** 2)

        step = 0.0
        if gradient_along_momentum > 0 and curvature > 0:
            step = gradient_along_momentum / curvature
            self.W -= step * self.vW
            self.b -= step * self.vb

        return step
