import numpy as np

from .AbstractKNN import KNN


class SimpleKNN(KNN):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def w(self, i):
        return np.int32(i < self.k)

    def predict_prob(self, x):

        dist_list = np.linalg.norm(self.X - x, axis=1)
        idx_sort = np.argsort(dist_list)

        idx_list = np.arange(self.X.shape[0])

        weights = self.w(idx_list)[:, np.newaxis]

        pre_class_sum = np.sum(self.y[idx_sort] * weights, axis=0)
        return pre_class_sum / pre_class_sum.sum()
