import gzip
import io
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/meet3010/linear-regression-on-the-diabetes-dataset"
DATASET_PATH = Path(__file__).resolve().parent / "data" / "diabetes_regression.csv"
FEATURE_COLUMNS = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
TARGET_COLUMN = "target"


SKLEARN_DATA_URL = (
    "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.8.0/"
    "sklearn/datasets/data"
)


def _validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    expected = (*FEATURE_COLUMNS, TARGET_COLUMN)

    for column in expected:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains missing or non-finite values")

    return df


def _download_dataset(timeout: int = 10) -> pd.DataFrame:
    def read_gzip(filename: str) -> np.ndarray:
        with urlopen(f"{SKLEARN_DATA_URL}/{filename}", timeout=timeout) as response:
            compressed = response.read()
        return np.loadtxt(io.BytesIO(gzip.decompress(compressed)), dtype=float)

    X = read_gzip("diabetes_data_raw.csv.gz")
    y = read_gzip("diabetes_target.csv.gz")

    X = (X - X.mean(axis=0)) / X.std(axis=0) / np.sqrt(X.shape[0])
    return pd.DataFrame(np.column_stack((X, y)), columns=[*FEATURE_COLUMNS, TARGET_COLUMN])

def _load_bundled_dataset() -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    return dataset.frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()

def load_dataset(
    output_path: str | Path | None = None,
    source_path: str | Path = DATASET_PATH,
) -> pd.DataFrame:

    source = Path(source_path)
    destination = Path(output_path) if output_path is not None else source

    if destination.is_file():
        return _validate_dataset(pd.read_csv(destination))

    if source.is_file():
        df = _validate_dataset(pd.read_csv(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return df

    try:
        df = _validate_dataset(_download_dataset())
    except (OSError, EOFError, ValueError):
        df = _validate_dataset(_load_bundled_dataset())

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".csv.tmp")

    try:
        df.to_csv(temporary_path, index=False)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return df


def analyze_dataset(df: pd.DataFrame, output_path: str | Path | None = None) -> None:

    print("\nDataset preview:\n", df.head().to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nDescriptive statistics:\n", df.describe().to_string())

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.describe().to_csv(path)

def train_val_test_split(
    df: pd.DataFrame,
    target_column: str = "target",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df))
    n_test = round(len(df) * test_size)
    n_val = round(len(df) * val_size)

    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    X = df.drop(columns=[target_column]).to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    return (
        X[train_indices], X[val_indices], X[test_indices],
        y[train_indices], y[val_indices], y[test_indices],
    )

class StandardScaler:

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.means = X.mean(axis=0)
        stds = X.std(axis=0)
        self.stds = np.where(stds < self.eps, 1.0, stds)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
