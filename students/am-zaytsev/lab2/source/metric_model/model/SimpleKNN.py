import numpy as np


class SimpleKNN:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k

    def train(self, X, y):
        self.X = X
        self.y = y

    def w(self, i):
        return np.int32(i < self.k)

    def predict_prob(self, x):

        dist_list = np.linalg.norm(self.X - x, axis=1)
        idx_sort = np.argsort(dist_list)

        idx_list = np.arange(self.X.shape[0])

        weights = self.w(idx_list)[:, np.newaxis]

        pre_class_sum = np.sum(self.y[idx_sort] * weights, axis=0)
        return pre_class_sum / pre_class_sum.sum()

    def predict_class(self, x):
        class_distrib = self.predict_prob(x)
        return np.argmax(class_distrib)
