import numpy as np

from .Model import Model


class LinearClassifier(Model):
    def __init__(self, feat_data, target_data):

        self.x = feat_data
        self.y = target_data
        self.init_weights()

    def _add_ones(self, x):
        ones = np.ones(x.shape[0])[:, np.newaxis]
        return np.concat([x, ones], 1)

    def init_weights(self):
        n = self.x.shape[0]
        self.w = np.random.uniform(-1 / (2 * n), 1 / (2 * n), self.x.shape[1] + 1)[
            :, np.newaxis
        ]

    def predict_margin(self, feat_data, target_data, need_pad=False):
        if need_pad:
            feat_data = self._add_ones(feat_data)
        return feat_data.dot(self.w) * target_data

    def loss(self, feat_data, target_data):
        feat_data = self._add_ones(feat_data)

        return (
            1 - (self.predict_margin(feat_data, target_data)).mean()
        ) ** 2  # + self.w.dot(self.w.T)

    def predict(self, feat_data):
        feat_data = self._add_ones(feat_data)
        return (feat_data.dot(self.w) > 0) * 2 - 1

    def _diff(self, feat_data, target_data, tao):
        feat_data = self._add_ones(feat_data)
        n = feat_data.shape[0]

        pred = feat_data.dot(self.w)  # X @ A
        residual = 1 - target_data * pred  # 1 - Y ⊙ (X @ A)

        grad = -2 / n * feat_data.T.dot(target_data * residual) + tao * self.w
        return grad

    def diff(self, feat_data, target_data, tao):
        if feat_data.shape[1] != self.w.shape[0]:
            feat_data = self._add_ones(feat_data)
        n = feat_data.shape[0]
        c1 = -2 / n * feat_data.T.dot(target_data)
        c2 = 2 / n * feat_data.T.dot(target_data**2 * feat_data) + tao * np.eye(
            self.w.shape[0]
        )

        dL = c1 + c2.dot(self.w)
        return dL, c1, c2
