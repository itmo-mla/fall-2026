import numpy as np

from .AbstractKNN import KNN


class SimpleKNN(KNN):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def w(self, i):
        return np.int32(i < self.k)

    def predict_prob(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        x = np.atleast_2d(x)

        dist_list = np.linalg.norm(self.X[None, :, :] - x[:, None, :], axis=2)
        idx_sort = np.argsort(dist_list, axis=1)

        idx_list = np.arange(self.X.shape[0])

        weights = self.w(idx_list)[np.newaxis, :, np.newaxis]

        pre_class_sum = np.sum(self.y[idx_sort] * weights, axis=1)

        total = pre_class_sum.sum(axis=1, keepdims=True)
        n_classes = self.y.shape[1]
        probs = np.divide(
            pre_class_sum,
            total,
            out=np.full_like(pre_class_sum, 1.0 / n_classes, dtype=float),
            where=total != 0,
        )
        return probs[0] if single else probs
