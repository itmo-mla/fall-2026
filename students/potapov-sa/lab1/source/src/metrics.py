from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from .dataset import DataSample, DSType

if TYPE_CHECKING:
    from .model import Model


def confusion_matrix(Y: np.ndarray, Y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp, tn, fp, fn = 0, 0, 0, 0
    for y, y_pred in zip(Y, Y_pred):
        if y == y_pred:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_pred == 1:
                fp += 1
            else:
                fn += 1
    return tp, tn, fp, fn


@dataclass
class SampleMetrics:
    sample_name: Optional[str]
    sample_type: Optional[DSType]
    accuracy: float
    precision: float
    recall: float

    @staticmethod
    def from_model(model: Model, sample: DataSample) -> SampleMetrics:
        Y_pred = model.predict(sample)
        tp, tn, fp, fn = confusion_matrix(sample.Y.to_numpy(), Y_pred)
        return SampleMetrics(
            sample_name=sample.name,
            sample_type=sample.type,
            accuracy=(tp + tn) / denum if (denum := tp + tn + fp + fn) != 0 else 0,
            precision=tp / denum if (denum := tp + fp) != 0 else 0,
            recall=tp / (tp + fn) if (denum := tp + fn) != 0 else 0
        )

    def to_dict(self) -> dict[str, float]:
        return {
            'Accuracy': self.accuracy,
            'Precision': self.precision,
            'Recall': self.recall
        }

    def __str__(self) -> str:
        title: list[str] = (
            [self.sample_type] if self.sample_type is not None else
            [self.sample_name] if self.sample_name is not None else []
        )
        return '\n'.join(
            title +
            [
                f'Accuracy: {self.accuracy}',
                f'Precision: {self.precision}',
                f'Recall: {self.recall}'
            ]
        )


@dataclass
class Metrics:
    train: Optional[SampleMetrics] = None
    test: Optional[SampleMetrics] = None

    def measure(self, model: 'Model', sample: DataSample, inplace: bool = True) -> SampleMetrics:
        if sample.type is None or not hasattr(self, sample.type):
            raise Exception(f'Cannot measure metrics for {sample.name}')

        sample_metrics = SampleMetrics.from_model(model, sample)
        if inplace:
            setattr(self, sample.type, SampleMetrics.from_model(model, sample))
        return sample_metrics
    

    def __str__(self) -> str:
        return '\n\n'.join(
            ([str(self.train)] if self.train is not None else []) +
            ([str(self.test)] if self.test is not None else [])
        )


__all__ = ['Metrics', 'SampleMetrics']
