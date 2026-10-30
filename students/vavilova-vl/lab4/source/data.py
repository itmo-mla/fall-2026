"""Загрузка и подготовка датасета Titanic для линейной регрессии."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
# Исходный датасет: https://www.kaggle.com/datasets/yasserh/titanic-dataset
KAGGLE_DATASET = "yasserh/titanic-dataset"


@dataclass
class Dataset:
    X: np.ndarray            # матрица «объекты–признаки», ℓ × n; пропуски — NaN
    y: np.ndarray            # целевая переменная, ℓ
    feature_names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


class Preprocessor:
    """Заполнение пропусков медианой и стандартизация."""

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.median_ = np.nanmedian(X, axis=0)
        X = self._impute(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._impute(X) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.median_, X)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Числовые признаки пассажира.

    """
    embarked = df["Embarked"]
    missing_port = embarked.isna()
    return pd.DataFrame({
        "Pclass": df["Pclass"],
        "Sex": (df["Sex"] == "female").astype(float),
        "Age": df["Age"],
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "HasCabin": df["Cabin"].notna().astype(float),
        "Embarked_C": (embarked == "C").astype(float).mask(missing_port),
        "Embarked_Q": (embarked == "Q").astype(float).mask(missing_port),
        "Survived": df["Survived"],
    })


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Скачивает CSV с Kaggle, если его ещё нет: в репозиторий *.csv не попадают.

    Датасет публичный, ключ Kaggle API не нужен. kagglehub кладёт файлы
    в свой кэш, оттуда CSV копируется рядом с кодом.
    """
    if not path.exists():
        print(f"Скачиваю датасет с Kaggle: {KAGGLE_DATASET}")
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        shutil.copy(cache_dir / path.name, path)
    return path


def load_dataset(path: Path = DATA_PATH) -> Dataset:
    """Titanic: 891 пассажир; цель — log(1 + Fare), стоимость билета в лог-шкале.

    Fare сильно скошен (медиана 14, максимум 512), логарифм делает зависимость
    ближе к линейной. Признаки возвращаются без заполнения пропусков и без
    стандартизации — см. Preprocessor.
    """
    df = pd.read_csv(download_dataset(path))
    features = build_features(df)
    X = features.to_numpy(dtype=float)
    y = np.log1p(df["Fare"].to_numpy(dtype=float))
    return Dataset(X=X, y=y, feature_names=list(features.columns))
