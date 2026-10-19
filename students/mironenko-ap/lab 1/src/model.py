"""Единый интерфейс линейного классификатора."""
import numpy as np
from .init import init_correlation, multistart, init_random
from .optim import sgd_momentum, steepest_descent


class LinearClassifier:
    def __init__(self, optimizer="momentum", ordering="random",
                 init="correlation", lam=1e-3, lr=0.01, mu=0.9,
                 epochs=50, multistart_n=None, seed=42,
                 print_every=10, verbose=True,
                 early_stop_delta=None, early_stop_patience=1):
        self.optimizer = optimizer
        self.ordering = ordering
        self.init = init
        self.lr = lr
        self.mu = mu
        self.epochs = epochs
        self.multistart_n = multistart_n
        self.seed = seed
        self.print_every = print_every
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.history = []

    def _opt(self):
        if self.optimizer == "momentum": #SGV
            return sgd_momentum
        if self.optimizer == "steepest": #Наискорейщий
            return steepest_descent
        raise ValueError(f"optimizer={self.optimizer}")

    def _fit_once(self, w0, b0, X, y, X_val=None, y_val=None, verbose=None):
        v = self.verbose if verbose is None else verbose
        fn = self._opt()
        kwargs = dict(
            use_l2=self.use_l2,
            epochs=self.epochs, verbose=v,
            print_every=self.print_every,
            ema_alphas=(0.9, 0.99, 0.999),
            X_val=X_val, y_val=y_val,
            early_stop_delta=self.early_stop_delta,
            early_stop_patience=self.early_stop_patience,
        )
        if self.optimizer == "momentum":
            kwargs.update(
                lr=self.lr, mu=self.mu, ordering=self.ordering,
                rng=np.random.default_rng(self.seed),
            )
        return fn(X, y, w0, b0, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):

        if self.init == "correlation":
            w0, b0 = init_correlation(X, y)
        elif self.init == "random":
            w0, b0 = init_random(X, y)
        else:
            raise ValueError(f"init={self.init}")

        if self.multistart_n:
            Xv = X_val if X_val is not None else X
            yv = y_val if y_val is not None else y
            self.w, self.b, self.history = multistart(
                self._fit_once, X, y, Xv, yv,
                n_starts=self.multistart_n, seed=self.seed, lam=self.lam,
            )
        else:
            self.w, self.b, self.history = self._fit_once(
                w0, b0, X, y, X_val=X_val, y_val=y_val,
            )
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)