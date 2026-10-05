import os
from typing import Optional
from io import StringIO

import pandas as pd
import numpy as np
# from kagglehub import dataset_download

from .constants import *


class PandasWrapper:
    _df: pd.DataFrame
    name: Optional[str]

    def __init__(self, name: Optional[str] = None):
        self.name = name

    def info(self) -> None:
        if self.name is not None:
            print(self.name)
        self._df.info()

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
        path = '~/.cache/kagglehub/datasets/mssmartypants/water-quality/versions/3'
        return pd.read_csv(os.path.join(path, DATASET_FILENAME))

    @staticmethod
    def _remove_num_err_with_zero(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
        for col in cols:
            df = df[df[col] != '#NUM!']
        return df

    def _prepare_dataset(self, target: str) -> pd.DataFrame:
        df = self._load_dataset()
        df = self._remove_num_err_with_zero(df, 'ammonia', target)
        df = df.astype({target: np.int16, 'ammonia': np.float64})
        df[target] -= 1 - df[target]
        return df

    def _train_test_split(self, target: str, ratio: float) -> tuple[DataSample, DataSample]:
        train_sample = self._df.sample(frac=ratio)
        test_sample = self._df.drop(train_sample.index)
        train_sample.reset_index(drop=True)
        test_sample.reset_index(drop=True)
        return (
            DataSample(train_sample, target, dataset_name=self.name, name='train'),
            DataSample(test_sample, target, dataset_name=self.name, name='test')
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
            dataset_name: Optional[str] = None,
            name: Optional[str] = None
    ):
        self.short_name = name
        if dataset_name is not None and name is not None:
            name = ': '.join([dataset_name, name])
        super().__init__(name=name)
        self._df = df
        self._X = df.drop([target], axis=1)
        self._X_1 = self._X.copy()
        if 'dummy' in self._X_1.columns:
            raise Exception('Invalid dataset column name: "dummy"')
        self._X_1['dummy'] = 1
        self._Y = df[target]
        self.name = name

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def X_1(self) -> pd.DataFrame:
        return self._X_1

    @property
    def Y(self) -> pd.Series:
        return self._Y

    def get_name(self, explicit: bool = False) -> Optional[str]:
        return self.name if explicit else self.short_name


__all__ = ['DataSet', 'DataSample']
