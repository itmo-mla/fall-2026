import numpy as np


def margin(w, X, y):
    return (X @ w) * y


def loss(M):
    return (1 - M) ** 2


def loss_grad(w, x, y):
    M = (x @ w) * y
    return -2 * (1 - M) * y * x


def steepest_step(x):
    return 1.0 / (2 * (x @ x))


class LinearClassifier:
    def __init__(self, lr=1e-3, momentum=0.9, tau=1e-3, lam=None, n_epochs=50,
                 step="const", order="random", init="random", temperature=1.0,
                 tol=1e-5, seed=None):
        self.lr = lr
        self.momentum = momentum
        self.tau = tau
        self.lam = lam
        self.n_epochs = n_epochs
        self.step = step
        self.order = order
        self.init = init
        self.temperature = temperature
        self.tol = tol
        self.seed = seed
        self.w = None
        self.history = None

    def init_weights(self, X, y, rng):
        n = X.shape[1]
        if self.init == "zeros":
            return np.zeros(n)
        if self.init == "random":
            return rng.uniform(-1 / (2 * n), 1 / (2 * n), n)
        if self.init == "correlation":
            return (y @ X) / np.sum(X ** 2, axis=0)
        raise ValueError(f"unknown init: {self.init}")

    def _epoch_order(self, w, X, y, rng):
        ell = len(y)
        if self.order == "random":
            return rng.permutation(ell)
        if self.order == "margin":
            abs_m = np.abs(margin(w, X, y))
            p = np.exp(-(abs_m - abs_m.min()) / self.temperature)
            return rng.choice(ell, size=ell, p=p / p.sum())
        raise ValueError(f"unknown order: {self.order}")

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        ell = len(y)
        lam = self.lam if self.lam is not None else 1 / ell

        w = self.init_weights(X, y, rng)
        v = np.zeros_like(w)
        subset = rng.choice(ell, size=min(ell, 50), replace=False)
        Q = loss(margin(w, X[subset], y[subset])).mean()

        history = {"Q": [], "Q_epoch": [], "train_acc": [], "val_acc": []}
        w_init = w.copy()
        for _ in range(self.n_epochs):
            Q_prev = Q
            for i in self._epoch_order(w, X, y, rng):
                x_i, y_i = X[i], y[i]
                xi = loss((x_i @ w) * y_i)
                g = loss_grad(w, x_i, y_i)
                if self.step == "steepest":
                    h, v = steepest_step(x_i), g
                else:
                    h = self.lr
                    v = self.momentum * v + (1 - self.momentum) * g
                w = w * (1 - h * self.tau) - h * v

                Q = lam * xi + (1 - lam) * Q
                history["Q"].append(Q)

            history["Q_epoch"].append(Q)
            history["train_acc"].append(np.mean(self._sign(X @ w) == y))
            if X_val is not None:
                history["val_acc"].append(np.mean(self._sign(X_val @ w) == y_val))
            if abs(Q - Q_prev) < self.tol:
                break

        self.w_init = w_init
        self.w = w
        self.history = history
        return self

    @staticmethod
    def _sign(z):
        return np.where(z >= 0, 1, -1)

    def decision_function(self, X):
        return X @ self.w

    def predict(self, X):
        return self._sign(self.decision_function(X))

    def empirical_risk(self, X, y):
        return loss(margin(self.w, X, y)).mean()


def multistart(X, y, n_starts=10, seed=0, X_val=None, y_val=None, **params):
    models = []
    for s in range(n_starts):
        clf = LinearClassifier(init="random", seed=seed + s, **params)
        models.append(clf.fit(X, y, X_val, y_val))
    risks = [m.empirical_risk(X, y) for m in models]
    return models[int(np.argmin(risks))], risks
