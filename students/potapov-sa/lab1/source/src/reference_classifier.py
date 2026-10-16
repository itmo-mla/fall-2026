from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier

from .dataset import DataSample

from .model import Model

class ReferenceSGDClassifier(Model):
    def __init__(self, *args: Any, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(name=name)
        self._clf = SGDClassifier(*args, **kwargs, loss='squared_hinge', penalty='l2')

    def _fit_impl(self, sample: DataSample, **kwargs: Any) -> None:
        self._clf.fit(sample.X, sample.Y, **kwargs)
    
    def predict(self, sample: DataSample):
        return self._clf.predict(sample.X)

    def get_sample_X(self, sample: DataSample) -> pd.DataFrame:
        return sample.X
    
    @property
    def weights(self) -> np.ndarray:
        return np.concat([self._clf.coef_.flatten(), self._clf.intercept_])
