from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/rajyellow46/wine-quality"
)
DATASET_SHA256 = "cb01f91c70de066f85d4cffac3bb163e4719523861cd79ba8825b722cf98b48c"


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray


@dataclass
class PreparedData:
    datasets: dict[str, Dataset]
    cleaning_summary: pd.DataFrame
    distribution: pd.DataFrame
    split_summary: pd.DataFrame


def load_data() -> pd.DataFrame:
    request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        zip_contents = response.read()

    with ZipFile(BytesIO(zip_contents)) as archive:
        csv_contents = archive.read("winequalityN.csv")

    if sha256(csv_contents).hexdigest() != DATASET_SHA256:
        raise ValueError("Датасет изменился: SHA-256 не совпадает")

    return pd.read_csv(BytesIO(csv_contents))


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    without_missing = data.dropna()
    without_duplicates = without_missing.drop_duplicates()
    return without_duplicates.reset_index(drop=True)


def summarize_cleaning(
    original_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
) -> pd.DataFrame:
    original_count = len(original_data)
    missing_count = int(original_data.isna().any(axis=1).sum())
    duplicate_count = original_count - missing_count - len(cleaned_data)
    return pd.DataFrame(
        {"Количество": [original_count, missing_count, duplicate_count, len(cleaned_data)]},
        index=["Исходные строки", "Пропуски", "Повторы", "После очистки"],
    )


def class_distribution(data: pd.DataFrame) -> pd.DataFrame:
    counts = data["type"].value_counts().reindex(["white", "red"])
    fractions = counts / len(data)
    return pd.DataFrame({"count": counts, "fraction": fractions})


def split_grouped_data(
    data: pd.DataFrame,
    second_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = data.drop(columns="type")

    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=second_size,
        random_state=seed,
    )
    first_indices, second_indices = next(splitter.split(data, groups=groups))
    return data.iloc[first_indices], data.iloc[second_indices]


def split_train_validation_test(
    data: pd.DataFrame,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    train_data, remaining_data = split_grouped_data(data, second_size=0.4, seed=seed)
    validation_data, test_data = split_grouped_data(
        remaining_data, second_size=0.5, seed=seed + 1
    )
    return {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }


def standardize_splits(
    data_splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    scaler = StandardScaler()
    training_features = data_splits["train"].drop(columns="type")

    scaler.fit(training_features)

    standardized_splits = {}
    for name, part in data_splits.items():
        features = part.drop(columns="type")
        scaled_features = scaler.transform(features)
        assert np.isfinite(scaled_features).all()
        standardized_data = pd.DataFrame(
            scaled_features, columns=features.columns, index=part.index
        )
        standardized_data["type"] = part["type"]
        standardized_splits[name] = standardized_data
    return standardized_splits


def add_bias_feature(features: np.ndarray) -> np.ndarray:
    bias_column = -np.ones(len(features))
    return np.column_stack([bias_column, features])


def build_dataset(data: pd.DataFrame) -> Dataset:
    features = data.drop(columns="type").to_numpy(dtype=float)
    labels = data["type"].map({"white": -1, "red": 1}).to_numpy(dtype=float)
    return Dataset(features=add_bias_feature(features), labels=labels)


def summarize_splits(datasets: dict[str, Dataset]) -> pd.DataFrame:
    rows = []
    for name, dataset in datasets.items():
        rows.append({
            "split": name,
            "objects": len(dataset.labels),
            "red_fraction": float((dataset.labels == 1).mean()),
        })
    return pd.DataFrame(rows)


def prepare_data(seed: int = 42) -> PreparedData:
    raw_data = load_data()
    classification_data = raw_data.drop(columns="quality")
    cleaned_data = clean_data(classification_data)

    data_splits = split_train_validation_test(cleaned_data, seed)
    standardized_splits = standardize_splits(data_splits)
    datasets = {
        name: build_dataset(part)
        for name, part in standardized_splits.items()
    }

    return PreparedData(
        datasets=datasets,
        cleaning_summary=summarize_cleaning(classification_data, cleaned_data),
        distribution=class_distribution(cleaned_data),
        split_summary=summarize_splits(datasets),
    )
