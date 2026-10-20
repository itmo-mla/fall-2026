from dataclasses import dataclass
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "charitarth/pulsar-dataset-htru2"
)
FEATURE_NAMES = [
    "profile_mean", "profile_std", "profile_kurtosis", "profile_skewness",
    "dm_snr_mean", "dm_snr_std", "dm_snr_kurtosis", "dm_snr_skewness",
]


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray


@dataclass
class PreparedData:
    train: Dataset
    test: Dataset


def load_data() -> pd.DataFrame:
    request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        archive_contents = response.read()
    with ZipFile(BytesIO(archive_contents)) as archive:
        with archive.open("HTRU_2.csv") as csv_file:
            return pd.read_csv(csv_file, names=FEATURE_NAMES + ["label"])


def add_bias_feature(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features])


def prepare_data(raw_data: pd.DataFrame) -> PreparedData:
    features = raw_data[FEATURE_NAMES].to_numpy(dtype=float)
    labels = raw_data["label"].to_numpy(dtype=int)
    training_features, test_features, training_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    scaler = StandardScaler().fit(training_features)
    return PreparedData(
        train=Dataset(
            features=add_bias_feature(scaler.transform(training_features)),
            labels=training_labels,
        ),
        test=Dataset(
            features=add_bias_feature(scaler.transform(test_features)),
            labels=test_labels,
        ),
    )
