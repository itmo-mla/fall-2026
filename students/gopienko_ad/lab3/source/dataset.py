from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


COLUMNS = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class",
]

COLUMNS_TO_IMPUTE = [
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
]


def load_dataset(
    output_dir: str | Path = "data",
) -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download(
        "kumargh/pimaindiansdiabetescsv",
        output_dir=str(output_dir),
    )

    dataset_path = Path(dataset_path)
    csv_files = list(dataset_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"CSV file was not found in {dataset_path}."
        )

    df = pd.read_csv(
        csv_files[0],
        names=COLUMNS,
        header=None,
    )

    if len(df) > 0 and str(df.iloc[0]["class"]).lower() == "class":
        df = df.iloc[1:].reset_index(drop=True)

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["class"] = df["class"].astype(int)

    return df


def analyze_dataset(
    df: pd.DataFrame,
) -> None:
    print("First rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nStatistics:")
    print(df.describe())

    print("\nZero values in columns where zero means missing:")
    print((df[COLUMNS_TO_IMPUTE] == 0).sum())

    print("\nClass distribution:")
    print(df["class"].value_counts().sort_index())


def calculate_impute_medians(
    df: pd.DataFrame,
) -> dict[str, float]:
    medians = {}

    for column in COLUMNS_TO_IMPUTE:
        values = df.loc[df[column] != 0, column]
        medians[column] = float(values.median())

    return medians


def impute_zero_values(
    df: pd.DataFrame,
    medians: dict[str, float],
) -> pd.DataFrame:
    df = df.copy().astype(float)

    for column in COLUMNS_TO_IMPUTE:
        df.loc[
            df[column] == 0,
            column,
        ] = medians[column]

    return df


def train_val_test_split(
    df: pd.DataFrame,
    target_column: str = "class",
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
):
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative.")

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

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

        y_numpy = y.to_numpy()

        for cls in np.unique(y_numpy):
            cls_indices = np.flatnonzero(y_numpy == cls)
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
        X.iloc[train_indices].reset_index(drop=True),
        X.iloc[val_indices].reset_index(drop=True),
        X.iloc[test_indices].reset_index(drop=True),

        y.iloc[train_indices].to_numpy(),
        y.iloc[val_indices].to_numpy(),
        y.iloc[test_indices].to_numpy(),
    )


class StandardScaler:
    def __init__(
        self,
        eps: float = 1e-12,
    ):
        self.means = None
        self.stds = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)

        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        self.stds = np.where(
            self.stds < self.eps,
            1.0,
            self.stds,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("StandardScaler must be fitted before transform.")

        x = np.asarray(x, dtype=float)

        return (x - self.means) / self.stds

    def fit_transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        self.fit(x)

        return self.transform(x)
