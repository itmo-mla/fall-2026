from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "Survived"
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
LABEL_NAMES = {0: "Погиб", 1: "Выжил"}


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _encode(data: pd.DataFrame, age_fill: float, embarked_fill: str) -> np.ndarray:
    """Turn raw passenger records into numeric features."""
    embarked = data["Embarked"].fillna(embarked_fill)
    features = pd.DataFrame(
        {
            "Pclass": data["Pclass"],
            "Sex": (data["Sex"] == "female").astype(float),
            "Age": data["Age"].fillna(age_fill),
            "SibSp": data["SibSp"],
            "Parch": data["Parch"],
            # Fare is heavily right-skewed, the log keeps rich outliers from dominating distances.
            "Fare": np.log1p(data["Fare"]),
            "Embarked_C": (embarked == "C").astype(float),
            "Embarked_Q": (embarked == "Q").astype(float),
        }
    )
    return features[FEATURES].to_numpy(dtype=float)


def prepare_data(test_size: float = 0.2, random_state: int = 42) -> PreparedData:
    """Load, clean, split, and standardize the Titanic data without leakage."""
    data = pd.read_csv(Path(__file__).with_name("Titanic-Dataset.csv"))
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET],
    )

    # Imputation statistics come from the training part only.
    age_fill = float(train["Age"].median())
    embarked_fill = str(train["Embarked"].mode()[0])
    X_train = _encode(train, age_fill, embarked_fill)
    X_test = _encode(test, age_fill, embarked_fill)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return PreparedData(
        X_train,
        X_test,
        train[TARGET].to_numpy(dtype=int),
        test[TARGET].to_numpy(dtype=int),
        FEATURES.copy(),
    )


def main() -> None:
    data = prepare_data()
    print(f"Train: {data.X_train.shape}, test: {data.X_test.shape}")
    print("Train class counts:", dict(zip(*np.unique(data.y_train, return_counts=True))))
    print("Test class counts:", dict(zip(*np.unique(data.y_test, return_counts=True))))


if __name__ == "__main__":
    main()
