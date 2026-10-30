from typing import Optional, Literal
from io import StringIO
from kagglehub import load_dataset, KaggleDatasetAdapter

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
        return load_dataset(
            KaggleDatasetAdapter.PANDAS,
            'mayurdalvi/sonar-mine-dataset',
            'sonar data.csv'
        )

    def _prepare_dataset(self, target: str) -> pd.DataFrame:
        df = self._load_dataset()
        df[target] = df[target] == 'M'
        df = df.astype({target: np.int16})
        df[target] -= 1 - df[target]
        return df

    def _train_test_split(self, target: str, ratio: float) -> tuple[DataSample, DataSample]:
        train_sample = self._df.sample(frac=ratio)
        test_sample = self._df.drop(train_sample.index)
        train_sample.reset_index(drop=True)
        test_sample.reset_index(drop=True)
        train_sample, test_sample = self._scale_samples(train_sample, test_sample, target)
        return (
            DataSample(train_sample, target, dataset_name=self.name, type_='train'),
            DataSample(test_sample, target, dataset_name=self.name, type_='test')
        )

    def _scale_samples(
        self, train: pd.DataFrame, test: pd.DataFrame, *ignore: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert (train.columns == test.columns).all()
        for col in train.columns:
            if col in ignore:
                continue
            train_min = train[col].min()
            train_max = train[col].max()
            train[col] = (train[col] - train_min) / (train_max - train_min)
            test[col] = (test[col] - train_min) / (train_max - train_min)
        return train, test

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
        # self._df = self._scale(df, target)
        self._df = df
        self._X = self._df.drop([target], axis=1)
        self._X_1 = self._X.copy()
        if 'dummy' in self._X_1.columns:
            raise Exception('Invalid dataset column name: "dummy"')
        self._X_1['dummy'] = 1
        self._Y = df[target]

    def _scale(self, df: pd.DataFrame, *ignore: str) -> pd.DataFrame:
        for col in df.columns:
            if col in ignore:
                continue
            min_ = df[col].min()
            df[col] = (df[col] - min_) / (df[col].max() - min_)
        return df

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
