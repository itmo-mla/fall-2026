from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "atharvaingle/crop-recommendation-dataset"
)


def load_data() -> pd.DataFrame:
    request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        archive_contents = response.read()

    archive_buffer = BytesIO(archive_contents)
    with ZipFile(archive_buffer) as archive:
        csv_contents = archive.read("Crop_recommendation.csv")

    csv_buffer = BytesIO(csv_contents)
    return pd.read_csv(csv_buffer)


def prepare_data(
    raw_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cleaned_data = raw_data.dropna()
    cleaned_data = cleaned_data.drop_duplicates()
    features = cleaned_data.drop(columns="label").to_numpy(dtype=float)
    labels = cleaned_data["label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test
