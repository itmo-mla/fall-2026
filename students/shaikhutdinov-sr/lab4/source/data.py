from dataclasses import dataclass
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "elikplim/concrete-compressive-strength-data-set"
)
TARGET = "concrete_compressive_strength"


@dataclass
class PreparedData:
    train_features: np.ndarray
    test_features: np.ndarray
    train_targets: np.ndarray
    test_targets: np.ndarray
    feature_names: list[str]
    raw_count: int
    missing_count: int
    duplicate_count: int


def load_data() -> pd.DataFrame:
    request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        archive_contents = response.read()

    with ZipFile(BytesIO(archive_contents)) as archive:
        csv_contents = archive.read("concrete_data.csv")

    raw_data = pd.read_csv(BytesIO(csv_contents))
    raw_data.columns = raw_data.columns.str.strip()
    return raw_data


def prepare_data(raw_data: pd.DataFrame) -> PreparedData:
    complete_data = raw_data.dropna()
    cleaned_data = complete_data.drop_duplicates()
    feature_names = cleaned_data.drop(columns=TARGET).columns.tolist()
    features = cleaned_data[feature_names].to_numpy(dtype=float)
    targets = cleaned_data[TARGET].to_numpy(dtype=float)
    _, feature_groups = np.unique(features, axis=0, return_inverse=True)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(split.split(features, groups=feature_groups))
    train_features = features[train_indices]
    test_features = features[test_indices]

    scaler = StandardScaler().fit(train_features)
    return PreparedData(
        train_features=scaler.transform(train_features),
        test_features=scaler.transform(test_features),
        train_targets=targets[train_indices],
        test_targets=targets[test_indices],
        feature_names=feature_names,
        raw_count=len(raw_data),
        missing_count=len(raw_data) - len(complete_data),
        duplicate_count=len(complete_data) - len(cleaned_data),
    )
