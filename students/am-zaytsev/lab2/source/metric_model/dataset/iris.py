import os

import kagglehub
import pandas as pd

from .utils import draw_data_by_target, one_hot_encode

# Column holding the class label — adjust if your CSV uses a different name
LABEL_COL = "Species"


def download_dataset() -> pd.DataFrame:
    # Download latest version
    data_path = kagglehub.dataset_download("uciml/iris")
    df = pd.read_csv(os.path.join(data_path, "Iris.csv"))
    return df


def preprocess_data(df):
    # Kaggle's Iris.csv has an extra "Id" column — drop it if present
    df = df.drop(columns=[c for c in ("Id", "id") if c in df.columns])

    X = df.drop(LABEL_COL, axis=1).to_numpy()
    Y = df[LABEL_COL].to_numpy()
    return X, one_hot_encode(Y)


def draw_data(df):
    return draw_data_by_target(df, LABEL_COL)
