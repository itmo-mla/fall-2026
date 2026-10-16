from typing import Optional, Literal, Any

import numpy as np
import pandas as pd

from .model import Model
from .dataset import DataSample
from .exceptions import InitializationException
from .constants import RANDOM_STATE


class LinearClassificator(Model):
    def __init__(
            self,
            feature_count: int,
            name: Optional[str] = None,
            initial_weights: Optional[np.ndarray] = None,
            init_weights_method: Literal['random', 'corr'] = 'random',
            corr_sample: Optional[DataSample] = None,
            object_selection_method: Literal['shuffling', 'unsure_first'] = 'unsure_first',
            # learning_rate: float = 0.001,
            learning_rate: float = 10e-5,
            sgd_forgetting_rate: float = 0.5,
            nag_rate: float = 10e-3,
            regularization_rate: float = 10e-3,
            epsilon: float = 10e-5,
        ):
        super().__init__(name=name)

        self.feature_count = feature_count
        self._weights: np.ndarray = (
            initial_weights if initial_weights is not None else
            self._init_weights(init_weights_method, corr_sample=corr_sample)
        )
        self.object_selection_method = object_selection_method
        if learning_rate <= 0:
            raise InitializationException('Learning rate should be positive')
        self.lr = learning_rate
        self.lmbd = sgd_forgetting_rate
        self.gamma = nag_rate
        self.tao = regularization_rate
        self.epsilon = epsilon

        self.target_classes = [-1, 1]
        self.shuffling_i = 0

    @staticmethod
    def from_file(path: str) -> LinearClassificator:
        weights = np.fromfile(path)
        if len(weights.shape) > 1 or len(weights) < 1:
            raise InitializationException('Weights must be a non-null vector')
        return LinearClassificator(feature_count=weights.shape[-1], initial_weights=weights)

    def _init_weights(self, method: str, corr_sample: Optional[DataSample] = None) -> np.ndarray:
        if method == 'random':
            return np.random.uniform(
                -(1 / (2 * self.feature_count)),
                1 / (2 * self.feature_count),
                self.feature_count
            )
        if method == 'corr':
            if corr_sample is None:
                raise Exception(
                    'X subsample should be provided in order to init weights using correlation'
                )
            X, Y = corr_sample.X_1.to_numpy(), corr_sample.Y.to_numpy()
            return np.array(
                [np.dot(Y, X[:, j]) / np.sum(X[:, j] ** 2) for j in range(self.feature_count)] # type: ignore
            )
        raise NotImplementedError

    @staticmethod
    def softmin(x: np.ndarray) -> np.ndarray:
        np_exp_x = np.exp(-x)
        return np_exp_x / np.sum(np_exp_x)

    def get_sorted_abs_margins_indices_sample(self, X: np.ndarray, Y: np.ndarray, size: int = 1) -> np.ndarray:
        sorted_margins = pd.Series(np.abs(self.calc_margins(X, Y))).sort_values()
        sorted_margins_proba = self.softmin(sorted_margins.to_numpy())
        index = np.random.choice(sorted_margins.index.to_numpy(), size=size, p=sorted_margins_proba)
        return index

    def loss(self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return (1 - self.calc_margins(X, Y, weights=weights)) ** 2

    @staticmethod
    def loss_margin(m: np.ndarray) -> np.ndarray:
        return (1 - m) ** 2

    def gradient_descent(self, sample: DataSample):
        X, Y = sample.X_1.to_numpy(), sample.Y.to_numpy()
        
        q = np.sum(self.loss(X, Y))
        delta_q = float('inf')

        while delta_q > self.epsilon:
            m = self.calc_margins(X, Y)
            grad = (-2 * Y * (1 - m) / X.shape[0]) @ X
            self._weights = self._weights * (1 - self.lr * self.tao) - self.lr * grad

            new_q = np.sum(self.loss(X, Y))
            delta_q = abs(q - new_q)
            q = new_q

    @staticmethod
    def _get_subsample(sample: DataSample, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        sub_X = sample.X_1.sample(frac=frac, random_state=RANDOM_STATE + 1)
        sub_Y = sample.Y[sub_X.index]
        sub_X.reset_index()
        sub_Y.reset_index()
        return sub_X.to_numpy(), sub_Y.to_numpy()

    def update_weights_nag(self, v: np.float64, x: np.ndarray, y: np.ndarray) -> np.float64:
        h = 1 / np.sum(x ** 2) ** 2 # steepest decent
        nag_weights = self._weights - h * self.gamma * v
        m = self.calc_margins(x, y, weights=nag_weights)
        grad = (-2 * y * (1 - m)) @ x
        v = (self.gamma * v + (1 - self.gamma) * grad)[0]
        self._weights = self._weights * (1 - h * self.tao) - h * v
        return v

    def select_object(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.object_selection_method == 'shuffling':
            cls = self.target_classes[self.shuffling_i]
            Y_ = pd.Series(Y)
            self.shuffling_i = (self.shuffling_i + 1) % len(self.target_classes)
            batch_rand_idx = Y_[Y_ == cls].sample(1).index.to_numpy()
        elif self.object_selection_method == 'unsure_first':
            batch_rand_idx = self.get_sorted_abs_margins_indices_sample(X, Y)
        else:
            raise NotImplementedError
        x, y = X[batch_rand_idx], Y[batch_rand_idx]
        return x, y

    def sgd(self, sample: DataSample, batch_size: int = 16):
        X, Y = sample.X_1.to_numpy(), sample.Y.to_numpy()

        q = np.mean(self.loss(*self._get_subsample(sample)))
        delta_q = float('inf')

        # NAG
        v = np.float64(0)
        
        i = 0
        while delta_q > self.epsilon:
            i += 1

            x, y = self.select_object(X, Y)

            m = self.calc_margins(x, y)
            xi = self.loss_margin(m)

            v = self.update_weights_nag(v, x, y)

            new_q = self.lmbd * xi + (1 - self.lmbd) * q
            # print(f'{i}. Q: {new_q}')
            delta_q = abs(q - new_q)
            q = new_q  

    def _fit_impl(self, sample: DataSample, **kwargs: Any) -> None:
        self.sgd(sample, **kwargs)
    
    def predict(self, sample: DataSample) -> np.ndarray:
        return np.array([1 if y_pred >= 0 else -1 for y_pred in self._weights @ sample.X_1.T])

    @property
    def weights(self) -> np.ndarray:
        return self._weights
