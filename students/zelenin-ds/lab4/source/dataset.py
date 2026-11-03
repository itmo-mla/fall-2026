from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

DATASET_HANDLE = "avikasliwal/used-cars-price-prediction"

TWO_WORD_BRANDS = {"Land": "Land Rover"}


def download_dataset() -> Path:
    path = kagglehub.dataset_download(DATASET_HANDLE)
    return Path(path) / "train-data.csv"


def extract_brand(name: str) -> str:
    brand = str(name).split(maxsplit=1)[0]
    return TWO_WORD_BRANDS.get(brand, brand)


def _parse_number_with_units(series: pd.Series) -> pd.Series:
    numbers = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
    values = pd.to_numeric(numbers, errors="coerce")
    return values.replace(0.0, np.nan)


def preprocess(df: pd.DataFrame, variance_clip_quantile: float = 0.995):
    df = df.copy()

    df = df.drop(columns=[c for c in ("Unnamed: 0", "New_Price") if c in df.columns])
    df = df.dropna(subset=["Price"])

    df["Brand"] = df["Name"].map(extract_brand)
    df = df.drop(columns=["Name"])

    for col in ("Mileage", "Engine", "Power"):
        df[col] = _parse_number_with_units(df[col])

    numeric_cols = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    q = df["Kilometers_Driven"].quantile(variance_clip_quantile)
    df["Kilometers_Driven"] = df["Kilometers_Driven"].clip(upper=q)

    categorical_cols = ["Brand", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
    df_num = pd.get_dummies(
        df[numeric_cols + categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=float,
    )

    X = df_num.to_numpy(dtype=float)
    y = df["Price"].to_numpy(dtype=float)
    return X, list(df_num.columns), y


def load_dataset(clip_quantile: float = 0.995):
    csv_path = download_dataset()
    df_raw = pd.read_csv(csv_path)
    X, feature_names, y = preprocess(df_raw, clip_quantile)
    return df_raw, X, feature_names, y
