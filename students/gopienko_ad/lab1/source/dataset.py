from pathlib import Path

import kagglehub
import pandas as pd
import numpy as np


COLUMNS = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class"
]


COLUMNS_TO_IMPUTE = [
    "plas",
    "pres",
    "skin",
    "test",
    "mass"
]


def load_dataset() -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download(
        "kumargh/pimaindiansdiabetescsv",
        output_dir="data"
    )

    dataset_path = Path(dataset_path)

    csv_path = next(dataset_path.glob("*.csv"))

    df = pd.read_csv(
        csv_path,
        names=COLUMNS,
        header=None
    )

    return df


def analyze_dataset(
        df: pd.DataFrame
) -> None:
    print("First rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nStatistics:")
    print(df.describe())

    print("\nZero values:")
    print((df[COLUMNS_TO_IMPUTE] == 0).sum())

    print("\nClass distribution:")
    print(df["class"].value_counts().sort_index())


def impute_zero_values(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    for column in COLUMNS_TO_IMPUTE:
        median = df.loc[
            df[column] != 0,
            column
        ].median()

        df.loc[
            df[column] == 0,
            column
        ] = median

    return df

def train_val_test_split(
        df: pd.DataFrame,
        target_column: str = "class",
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True,
        random_state: int = 42
):
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative.")

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.")

    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()

    rng = np.random.default_rng(random_state)

    if not stratify:
        indices = np.arange(len(df))
        rng.shuffle(indices)

        test_count = int(round(len(df) * test_size))
        val_count = int(round(len(df) * val_size))

        test_indices = indices[:test_count]
        val_indices = indices[test_count:test_count + val_count]
        train_indices = indices[test_count + val_count:]

    else:
        train_parts = []
        val_parts = []
        test_parts = []

        for df_cls in np.unique(y):
            cls_indices = np.flatnonzero(y == df_cls)
            rng.shuffle(cls_indices)

            test_count = int(round(len(cls_indices) * test_size))
            val_count = int(round(len(cls_indices) * val_size))

            test_parts.append(cls_indices[:test_count])
            val_parts.append(cls_indices[test_count:test_count + val_count])
            train_parts.append(cls_indices[test_count + val_count:])

        train_indices = np.concatenate(train_parts)
        val_indices = np.concatenate(val_parts)
        test_indices = np.concatenate(test_parts)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[val_indices],
        X[test_indices],
        y[train_indices],
        y[val_indices],
        y[test_indices],
    )


class StandardScaler:
    def __init__(self, eps: float = 1e-12):
        self.means = None
        self.stds = None 
        self.eps = eps 

    def fit(
            self,
            X: np.ndarray
    ) -> None:

        X = X.astype(float)

        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

        self.std = np.where(
            self.stds < self.eps,
            1.0,
            self.stds
        )

    def transform(self, X: np.ndarray) -> np.ndarray:

        if (self.means is None or self.stds is None):
            raise ValueError(
                "StandardScaler must be fitted before transform."
            )

        X = X.astype(float)

        X_scaled = (X - self.means) / self.stds

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
