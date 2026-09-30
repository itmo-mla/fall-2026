import numpy as np

from .ParzenKNN import ParzenKNN


class AdaptiveParzenKNN(ParzenKNN):
    def __init__(self, k, kernel_name="gauss"):
        super().__init__(k, kernel_name)


    def predict_prob(self, x):
        dist_list = np.linalg.norm(self.X - x, axis=1)
        p = np.sort(dist_list)[self.h]
        weights = self.w(dist_list, p)[:, np.newaxis]
        pre_class_sum = np.sum(self.y * weights, axis=0)  # без сортировки

        total = pre_class_sum.sum()
        if total == 0:  # пустое окно
            return np.full(self.y.shape[1], 1.0 / self.y.shape[1])
        return pre_class_sum / total
