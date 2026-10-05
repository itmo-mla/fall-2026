from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "Titanic-Dataset.csv"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
]
CLASS_NAMES = {-1: "Погиб", 1: "Выжил"}


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray


def encode_features(frame: pd.DataFrame, age_median: float, embarked_mode: str) -> pd.DataFrame:
    embarked = frame["Embarked"].fillna(embarked_mode)
    encoded = pd.DataFrame(
        {
            "Pclass": frame["Pclass"].astype(float),
            "Sex": (frame["Sex"] == "female").astype(float),
            "Age": frame["Age"].fillna(age_median).astype(float),
            "SibSp": frame["SibSp"].astype(float),
            "Parch": frame["Parch"].astype(float),
            "Fare": np.log1p(frame["Fare"].astype(float)),
            "Embarked_C": (embarked == "C").astype(float),
            "Embarked_Q": (embarked == "Q").astype(float),
        },
        index=frame.index,
    )
    return encoded[FEATURES]


def load_dataset(path: Path = DATA_PATH, test_size: float = 0.2, random_state: int = 42) -> Dataset:
    frame = pd.read_csv(path)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame["Survived"],
        random_state=random_state,
    )

    # Все статистики считаются только по train, чтобы не было утечки из test.
    age_median = float(train_frame["Age"].median())
    embarked_mode = str(train_frame["Embarked"].mode().iloc[0])

    X_train = encode_features(train_frame, age_median, embarked_mode).to_numpy()
    X_test = encode_features(test_frame, age_median, embarked_mode).to_numpy()

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    # SVM работает с метками из {-1, +1}: -1 - погиб, +1 - выжил.
    y_train = np.where(train_frame["Survived"].to_numpy() == 1, 1, -1)
    y_test = np.where(test_frame["Survived"].to_numpy() == 1, 1, -1)

    return Dataset(
        X_train=(X_train - mean) / std,
        X_test=(X_test - mean) / std,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(FEATURES),
        mean=mean,
        std=std,
    )
