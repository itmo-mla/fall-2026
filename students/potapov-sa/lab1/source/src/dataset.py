import os
from typing import Optional, Literal
from io import StringIO

import pandas as pd
import numpy as np

from .constants import *


DSType = Literal['train', 'test']


class PandasWrapper:
    _df: pd.DataFrame
    _type: Optional[DSType]
    _name: Optional[str]

    def __init__(self, type_: Optional[DSType] = None, name: Optional[str] = None):
        self._type = type_
        self._name = (': '.join([name, type_]) if type_ is not None else name) if name is not None else None

    def info(self) -> None:
        if self.name is not None:
            print(self.name)
        self._df.info()

    @property
    def type(self) -> Optional[DSType]:
        return self._type

    @property
    def name(self) -> Optional[str]:
        return self._name

    def __str__(self) -> str:
        buf = StringIO()
        self._df.info(buf=buf)
        s = buf.getvalue().replace('<class \'pandas.DataFrame\'>\n', '', 1)
        return (f'# {self.name} #\n' if self.name is not None else '') + s


class DataSet(PandasWrapper):
    def __init__(self,
            target: str = DATASET_TARGET,
            train_test_ratio: float = 0.7,
            name: Optional[str] = DATASET_NAME
    ):
        super().__init__(name=name)
        self._df = self._prepare_dataset(target)
        self.train, self.test = self._train_test_split(target, ratio=train_test_ratio)

    @staticmethod
    def _load_dataset() -> pd.DataFrame:
        path = '~/.cache/kagglehub/datasets/mayurdalvi/sonar-mine-dataset/versions/1'
        return pd.read_csv(os.path.join(path, DATASET_FILENAME))

    @staticmethod
    def _scale_dataset(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            min_ = df[col].min()
            df[col] = (df[col] - min_) / (df[col].max() - min_)
        return df

    def _prepare_dataset(self, target: str) -> pd.DataFrame:
        df = self._load_dataset()
        df[target] = df[target] == 'M'
        df = df.astype({target: np.int16})
        df[target] -= 1 - df[target]
        cols_wo_target = df.columns.drop(target)
        df[cols_wo_target] = self._scale_dataset(df[cols_wo_target])

        return df

    def _train_test_split(self, target: str, ratio: float) -> tuple[DataSample, DataSample]:
        train_sample = self._df.sample(frac=ratio)
        test_sample = self._df.drop(train_sample.index)
        train_sample.reset_index(drop=True)
        test_sample.reset_index(drop=True)
        return (
            DataSample(train_sample, target, dataset_name=self.name, type_='train'),
            DataSample(test_sample, target, dataset_name=self.name, type_='test')
        )

    @property
    def feature_count(self) -> int:
        return self._df.shape[-1] # target is counted (+1) instead of dummy feature (const 1)

    @property
    def df(self) -> pd.DataFrame:
        return self._df


class DataSample(PandasWrapper):
    def __init__(
            self,
            df: pd.DataFrame,
            target: str,
            type_: Literal['train', 'test'],
            dataset_name: Optional[str] = None
    ):
        super().__init__(type_=type_, name=dataset_name)
        self._df = df
        self._X = df.drop([target], axis=1)
        self._X_1 = self._X.copy()
        if 'dummy' in self._X_1.columns:
            raise Exception('Invalid dataset column name: "dummy"')
        self._X_1['dummy'] = 1
        self._Y = df[target]

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def X_1(self) -> pd.DataFrame:
        return self._X_1

    @property
    def Y(self) -> pd.Series:
        return self._Y


__all__ = ['DataSet', 'DataSample']
