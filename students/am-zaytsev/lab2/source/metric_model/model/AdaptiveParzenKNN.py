import numpy as np

from .ParzenKNN import ParzenKNN


class AdaptiveParzenKNN(ParzenKNN):
    def __init__(self, k, kernel_name="gauss"):
        super().__init__(k, kernel_name)


    def predict_prob(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        x = np.atleast_2d(x)

        dist_list = np.linalg.norm(self.X[None, :, :] - x[:, None, :], axis=2)
        p = np.sort(dist_list, axis=1)[:, self.h][:, np.newaxis]
        weights = self.w(dist_list, p)
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
