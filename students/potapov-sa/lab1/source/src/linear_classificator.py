from typing import Optional, Literal, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import Model
from .dataset import DataSample, DataSet
from .exceptions import InitializationException
from .constants import RANDOM_STATE


class LinearClassificator(Model):
    _loss_history_is_loggable = True

    def __init__(
            self,
            feature_count: int,
            name: Optional[str] = None,
            initial_weights: Optional[np.ndarray] = None,
            init_weights_method: Literal['random', 'corr'] = 'random',
            corr_sample: Optional[DataSample] = None,
            object_selection_method: Literal['shuffling', 'unsure_first'] = 'unsure_first',
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
        self.lmbd = sgd_forgetting_rate
        self.gamma = nag_rate
        self.tau = regularization_rate
        self.epsilon = epsilon

        self.target_classes = [-1, 1]
        self.shuffling_i = 0

    @staticmethod
    def from_file(path: str) -> 'LinearClassificator':
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

    def loss(
        self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        w = weights if weights is not None else self._weights
        return (1 - self.calc_margins(X, Y, weights=w)) ** 2 + self.tau / 2 * np.sum(
            w[:-1] ** 2
        )

    def loss_grad(
        self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
       w = weights if weights is not None else self._weights
       tau_w = np.concat([self.tau * w[:-1], np.array([0])])
       return (-2 * Y * (1 - self.calc_margins(X, Y, w))) @ X + tau_w


    @staticmethod
    def loss_margin(m: np.ndarray) -> np.ndarray:
        return (1 - m) ** 2

    @staticmethod
    def _get_subsample(sample: DataSample, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        sub_X = sample.X_1.sample(frac=frac, random_state=RANDOM_STATE + 1)
        sub_Y = sample.Y[sub_X.index]
        sub_X.reset_index()
        sub_Y.reset_index()
        return sub_X.to_numpy(), sub_Y.to_numpy()

    def _find_steepest_step(
        self, v: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.float64:
        w = self._weights
        res = minimize(
            lambda h: self.loss(
                x,
                y,
                w - h * (
                    self.gamma * v
                    + (1 - self.gamma) * self.loss_grad(x, y, w - h * self.gamma * v)
                ),
            ),
            x0=[0],
            bounds=[(0, None)],
        )
        return res.x[0]

    def update_weights_nag(self, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = self._find_steepest_step(v, x, y)
        grad = self.loss_grad(x, y, self._weights - h * self.gamma * v)
        v = self.gamma * v + (1 - self.gamma) * grad
        self._weights -= h * v
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

    def sgd(self, ds: DataSet, history: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        train_history: list[np.float64] = []
        test_history: list[np.float64] = []

        def _log_history():
            train_history.append(q)
            test_history.append(np.mean(self.loss(X_test, Y_test)))

        X_train, Y_train = ds.train.X_1.to_numpy(), ds.train.Y.to_numpy()
        X_test, Y_test = ds.test.X_1.to_numpy(), ds.test.Y.to_numpy()

        q = np.mean(self.loss(*self._get_subsample(ds.train)))
        delta_q = float('inf')
        delta_w = float('inf')

        # NAG
        v = np.zeros(self.feature_count)

        train_history: list[np.float64] = []
        test_history: list[np.float64] = []
        while delta_q > self.epsilon and delta_w > self.epsilon:
            old_weights = self._weights.copy()
            x, y = self.select_object(X_train, Y_train)

            m = self.calc_margins(x, y)
            xi = self.loss_margin(m)

            v = self.update_weights_nag(v, x, y)

            new_q = (self.lmbd * xi + (1 - self.lmbd) * q)[0] # new_q is a scalar
            delta_q = abs(q - new_q)

            delta_w = abs(np.sqrt(np.sum(self._weights ** 2)) - np.sqrt(np.sum(old_weights ** 2)))

            if history:
                _log_history()
            q = new_q

        if history:
            _log_history()
            return np.array(train_history), np.array(test_history)

    def _fit_impl(
        self, ds: DataSet, history: bool = False, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return self.sgd(ds, history=history, **kwargs)
    
    def predict(self, sample: DataSample) -> np.ndarray:
        return np.array([1 if y_pred >= 0 else -1 for y_pred in self._weights @ sample.X_1.T])

    @property
    def weights(self) -> np.ndarray:
        return self._weights
