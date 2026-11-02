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
        return self.kernel(dist_list / (h + 1e-6))

    def predict_prob(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        x = np.atleast_2d(x)

        dist_list = np.linalg.norm(self.X[None, :, :] - x[:, None, :], axis=2)
        weights = self.w(dist_list)
        pre_class_sum = weights @ self.y  # без сортировки

        total = pre_class_sum.sum(axis=1, keepdims=True)
        n_classes = self.y.shape[1]
        probs = np.divide(
            pre_class_sum,
            total,
            out=np.full_like(pre_class_sum, 1.0 / n_classes, dtype=float),
            where=total != 0,  # пустое окно
        )
        return probs[0] if single else probs
