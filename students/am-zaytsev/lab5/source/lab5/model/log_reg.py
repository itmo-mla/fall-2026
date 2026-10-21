import numpy as np


def check_shapes(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[0] == y.shape[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess_target(target):
    if len(target.shape) == 1:
        target = target[..., np.newaxis]
    if target.min() != -1:
        target = target * 2 - 1
    return target


class LogReg:
    def __init__(self):
        self._w = None

    def loss(self, x, y):
        margin = y * x @ self._w

        return np.log(1 + np.exp(-margin))

    def train(self, x, y, h=1, tol=0.01, max_iteration=1_000):
        preprocess_target(y)
        check_shapes(x, y)
        loss_list = []
        self._w = np.linalg.inv(x.T @ x) @ x.T @ y
        loss_value = self.loss(x, y)
        loss_list.append(loss_value)
        prev_sig = None
        for iter_id in range(max_iteration):
            margin = y * x @ self._w
            sig = sigmoid(margin)  # [..., np.newaxis]
            if prev_sig is not None:
                d_sig = np.mean(np.abs(prev_sig - sig) / np.abs(sig))
                if d_sig < tol:
                    break
            prev_sig = sig
            x_wave = x * sig
            y_star = np.sqrt((1 - sig) / sig) * y

            dw = -np.linalg.inv(x_wave.T @ x_wave) @ x_wave.T @ y_star
            self._w -= h * dw

            loss_value = self.loss(x, y)
            loss_list.append(loss_value)

        return loss_list

    def eval(self, x):
        return ((x @ self._w) > 0) * 2 - 1
