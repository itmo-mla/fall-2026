from dataclasses import dataclass
from typing import Optional, Any
from abc import ABC, abstractmethod

import numpy as np

from .dataset import DataSample, DataSet
from .metrics import Metrics, SampleMetrics


@dataclass
class LossHistory:
    train: np.ndarray
    test: np.ndarray


class Model(ABC):
    _metrics: Metrics
    _loss_history: Optional[LossHistory]
    _loss_history_is_loggable: bool = True

    def __init__(self, name: Optional[str] = None):
        self._name = name
        self._metrics = Metrics()
        self._loss_history = None

    def calc_margins_sample(
            self,
            sample: DataSample,
            weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X: np.ndarray = sample.X_1.to_numpy()
        Y: np.ndarray = sample.Y.to_numpy()
        return self.calc_margins(X, Y, weights=weights)

    def calc_margins(self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None):
        W: np.ndarray = weights if weights is not None else self.weights
        try:
            return (W @ X.T) * Y
        except Exception as e:
            print(self.__class__.__name__)
            raise e

    def measure_metrics(self, sample: DataSample, inplace: bool = True) -> SampleMetrics:
        return self._metrics.measure(self, sample, inplace=inplace)

    def fit(
        self, ds: DataSet, history: bool = True, evaluate: bool = True, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self._loss_history_is_loggable:
            kwargs['history'] = history
        else:
            history = False
        fit_res = self._fit_impl(ds, **kwargs)
        if evaluate:
            self.evaluate(ds.train)
        if history:
            self._loss_history = LossHistory(*fit_res) # type: ignore
        return fit_res

    def evaluate(self, sample: DataSample) -> Metrics:
        self._metrics.measure(self, sample)
        return self._metrics

    def save(self, filename: Optional[str] = None):
        self.weights.tofile(
            filename if filename else
            self.name if self.name else
            self.__class__.__name__
        )

    @property
    def name(self) -> str:
        return self._name if self._name else self.__class__.__name__

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def loss_history(self) -> Optional[LossHistory]:
        return self._loss_history

    @property
    def loss_history_is_loggable(self) -> bool:
        return self._loss_history_is_loggable

    # Abstract methods
    @abstractmethod
    def _fit_impl(
        self, ds: DataSet, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    @abstractmethod
    def predict(self, sample: DataSample) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        pass
