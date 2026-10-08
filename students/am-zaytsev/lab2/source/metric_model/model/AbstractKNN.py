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
        class_distrib = self.predict_prob(x)
        return np.argmax(class_distrib)
