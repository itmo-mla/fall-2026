import numpy as np

from .AbstractKNN import KNN
from .kernels import kernels_dict


class ParzenKNN(KNN):
    def __init__(self, h, kernel_name="gauss"):
        super().__init__()
        if kernel_name not in kernels_dict:
            raise KeyError(
                f"Invalid kernel_name: {kernel_name}. Available list: "
                + ", ".join(kernels_dict.keys())
            )
        self.h = h
        self.kernel = kernels_dict[kernel_name]

    def w(self, dist_list, h=None):
        if h is None:
            h = self.h
        return self.kernel(dist_list / h)

    def predict_prob(self, x):
        dist_list = np.linalg.norm(self.X - x, axis=1)
        weights = self.w(dist_list)[:, np.newaxis]
        pre_class_sum = np.sum(self.y * weights, axis=0)  # без сортировки

        total = pre_class_sum.sum()
        if total == 0:  # пустое окно
            return np.full(self.y.shape[1], 1.0 / self.y.shape[1])
        return pre_class_sum / total
