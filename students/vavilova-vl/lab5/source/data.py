"""Загрузка Titanic и подготовка признаков для логистической регрессии."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
# Исходный датасет: https://www.kaggle.com/datasets/yasserh/titanic-dataset
KAGGLE_DATASET = "yasserh/titanic-dataset"
TEST_SIZE = 0.2
SEED = 42


@dataclass
class Dataset:
    X_train: np.ndarray      # ℓ × n, стандартизованные признаки без константы
    y_train: np.ndarray      # метки {0, 1}: 1 — выжил
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def build_features(df: pd.DataFrame, age_median: float) -> pd.DataFrame:
    """Числовые признаки пассажира.

    Age заполняется медианой обучающей части, Embarked — самым частым портом S.
    Порт кодируется двумя индикаторами (S — базовый), иначе вместе с константой
    получилась бы точная линейная зависимость и вырожденный гессиан.
    Fare берётся в логарифме: распределение цены сильно скошено.
    """
    embarked = df["Embarked"].fillna("S")
    return pd.DataFrame({
        "Pclass": df["Pclass"],
        "Sex": (df["Sex"] == "female").astype(float),
        "Age": df["Age"].fillna(age_median),
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "LogFare": np.log1p(df["Fare"]),
        "HasCabin": df["Cabin"].notna().astype(float),
        "Embarked_C": (embarked == "C").astype(float),
        "Embarked_Q": (embarked == "Q").astype(float),
    })


def download_dataset(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        print(f"Скачиваю датасет с Kaggle: {KAGGLE_DATASET}")
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        shutil.copy(cache_dir / path.name, path)
    return path


def load_dataset(path: Path = DATA_PATH) -> Dataset:
    df = pd.read_csv(download_dataset(path))
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                         stratify=df["Survived"])
    age_median = train_df["Age"].median()
    train = build_features(train_df, age_median).to_numpy(dtype=float)
    test = build_features(test_df, age_median).to_numpy(dtype=float)

    # Статистики стандартизации — только по обучающей части.
    mean, std = train.mean(axis=0), train.std(axis=0)
    return Dataset(
        X_train=(train - mean) / std,
        y_train=train_df["Survived"].to_numpy(dtype=float),
        X_test=(test - mean) / std,
        y_test=test_df["Survived"].to_numpy(dtype=float),
        feature_names=list(build_features(df.head(1), age_median).columns),
    )


def add_bias(X: np.ndarray) -> np.ndarray:
    """F = [1, X]: константный признак f_0(x) = 1 даёт свободный член w_0."""
    return np.hstack([np.ones((X.shape[0], 1)), X])
