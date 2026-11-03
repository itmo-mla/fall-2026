from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "uciml/breast-cancer-wisconsin-data"
)


def load_data() -> pd.DataFrame:
    request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        archive_contents = response.read()

    with ZipFile(BytesIO(archive_contents)) as archive:
        csv_contents = archive.read("data.csv")
    return pd.read_csv(BytesIO(csv_contents))


def build_dataset(raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = raw_data.drop(columns=["id", "diagnosis", "Unnamed: 32"])
    labels = raw_data["diagnosis"].map({"B": -1, "M": 1})
    return features.to_numpy(dtype=float), labels.to_numpy(dtype=int)


def standardize_features(
    training_features: np.ndarray,
    evaluation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(training_features)
    scaled_evaluation = scaler.transform(evaluation_features)
    return scaled_training, scaled_evaluation


def prepare_validation_split(
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_features, validation_features, fit_labels, validation_labels = train_test_split(
        training_features,
        training_labels,
        test_size=0.25,
        random_state=42,
        stratify=training_labels,
    )
    fit_features, validation_features = standardize_features(
        fit_features, validation_features
    )
    return fit_features, validation_features, fit_labels, validation_labels
