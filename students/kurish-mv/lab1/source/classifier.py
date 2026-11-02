import numpy as np

from initialization import correlation_initialization, random_initialization
from loss import objective
from margin import calculate_margin


class LinearClassifier:
    def __init__(self, optimizer=None, initialization="random", seed=0, l2=None):
        self.optimizer = optimizer
        self.initialization = initialization
        self.seed = seed
        self.l2 = l2 if l2 is not None else getattr(optimizer, "l2", 0.0)
        self.weights, self.bias, self.history = None, 0.0, None

    def fit(self, X, y):
        if self.initialization == "random":
            w0, b0 = random_initialization(X.shape[1], self.seed)
        elif self.initialization == "correlation":
            w0, b0 = correlation_initialization(X, y)
        else:
            raise ValueError(f"Неизвестная инициализация: {self.initialization}")
        self.weights, self.bias, self.history = self.optimizer.fit(X, y, w0, b0)
        return self

    def decision_function(self, X):
        if self.weights is None:
            raise RuntimeError("Сначала вызовите fit()")
        return X @ self.weights + self.bias

    def predict(self, X, threshold=0.0):
        return np.where(self.decision_function(X) >= threshold, 1, -1)

    def margin(self, X, y):
        return calculate_margin(X, y, self.weights, self.bias)

    def objective(self, X, y):
        return objective(X, y, self.weights, self.bias, self.l2)
