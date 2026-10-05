from typing import Optional

import numpy as np

from .dataset import DataSample
from .exceptions import InitializationException
from .constants import RANDOM_STATE


class LinearClassificator:
    def __init__(
            self,
            feature_count: int,
            initial_weights: Optional[np.ndarray] = None,
            learning_rate: float = 0.001,
            sgd_forgetting_rate: float = 0.001,
            regularization_rate: float = 0.001
        ):
        self.feature_count = feature_count
        self.weights: np.ndarray = initial_weights if initial_weights is not None else self._init_weights() # vector
        if learning_rate <= 0:
            raise InitializationException('Learning rate should be positive')
        self.lr = learning_rate
        self.lmbd = sgd_forgetting_rate
        self.tao = regularization_rate

    @staticmethod
    def from_file(path: str) -> LinearClassificator:
        weights = np.fromfile(path)
        if len(weights.shape) > 1 or len(weights) < 1:
            raise InitializationException('Weights must be a non-null vector')
        return LinearClassificator(feature_count=weights.shape[-1], initial_weights=weights)

    def _init_weights(self) -> np.ndarray:
        return np.random.uniform(0, 1, self.feature_count)

    def calc_margins(self, X: np.ndarray, Y: np.ndarray):
        return (self.weights @ X.T) * Y

    def loss(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (1 - self.calc_margins(X, Y)) ** 2

    def gradient_descent(self, sample: DataSample):
        X, Y = sample.X_1.to_numpy(), sample.Y.to_numpy()
        
        q = np.sum(self.loss(X, Y))
        delta_q = float('inf')

        while delta_q > 10e-5:
            m = self.calc_margins(X, Y)
            grad = (-2 * Y * (1 - m) / X.shape[0]) @ X
            self.weights = self.weights * (1 - self.lr * self.tao) - self.lr * grad

            new_q = np.sum(self.loss(X, Y))
            delta_q = abs(q - new_q)
            q = new_q

    @staticmethod
    def _get_subsample(sample: DataSample, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        sub_X = sample.X_1.sample(frac=frac, random_state=RANDOM_STATE + 1)
        sub_Y = sample.Y[sub_X.index]
        print('sub_X:', sub_X, sep='\n', end='\n\n')
        print('sub_Y:', sub_Y, sep='\n')
        sub_X.reset_index()
        sub_Y.reset_index()
        return sub_X.to_numpy(), sub_Y.to_numpy()

    def sgd(self, sample: DataSample):
        X, Y = sample.X_1.to_numpy(), sample.Y.to_numpy()

        q = np.mean(self.loss(X, Y))
        print(f'{q=}')

        sub_X, sub_Y = self._get_subsample(sample)
        q = np.mean(self.loss(sub_X, sub_Y))
        print(f'{q=}')

    def predict(self, sample: DataSample) -> np.ndarray:
        return np.array([1 if y_pred >= 0 else -1 for y_pred in self.weights @ sample.X_1.T])

    def save(self, filename: str):
        self.weights.tofile(filename)
