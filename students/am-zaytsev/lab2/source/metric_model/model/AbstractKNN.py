from abc import ABC, abstractmethod

import numpy as np


class KNN(ABC):
    def __init__(self):
        self.X = None
        self.y = None

    def train(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def w(self, i):
        pass

    @abstractmethod
    def predict_prob(self, x):
        pass

    def predict_class(self, x):
        x = np.asarray(x)
        class_distrib = self.predict_prob(x)
        if x.ndim == 1:
            return np.argmax(class_distrib)
        return np.argmax(class_distrib, axis=-1)

    def acc(self, x, y_true):
        y_pred = self.predict_class(x)
        y_true = np.asarray(y_true)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        return (y_pred == y_true).mean()
