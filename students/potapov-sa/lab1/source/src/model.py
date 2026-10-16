from typing import Optional, Any
from abc import ABC, abstractmethod

import numpy as np

from .dataset import DataSample
from .metrics import Metrics, SampleMetrics


class Model(ABC):
    _margins: Optional[np.ndarray]
    _metrics: Metrics

    def __init__(self, name: Optional[str] = None):
        self._name = name
        self._metrics = Metrics()
        self._margins = None

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

    def fit(self, sample: DataSample, evaluate: bool = True, **kwargs: Any) -> None:
        self._fit_impl(sample, **kwargs)
        if evaluate:
            self.evaluate(sample)

    def evaluate(self, sample: DataSample) -> Metrics:
        self._margins = self.calc_margins_sample(sample)
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
    def margins(self) -> Optional[np.ndarray]:
        return self._margins

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    # Abstract methods
    @abstractmethod
    def _fit_impl(self, sample: DataSample, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(self, sample: DataSample) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        pass
