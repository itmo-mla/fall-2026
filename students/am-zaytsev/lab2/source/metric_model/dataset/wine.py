import os

import kagglehub
import pandas as pd

from .utils import one_hot_encode


def download_dataset() -> pd.DataFrame:
    data_path = kagglehub.dataset_download("yasserh/wine-quality-dataset")

    df = pd.read_csv(os.path.join(data_path, "WineQT.csv"))
    df["quality"] = df["quality"].clip(5, 7)
    return df


def preprocess_data(df):
    X = df.drop("quality", axis=1).to_numpy()
    Y = df["quality"].to_numpy()
    return X, one_hot_encode(Y)
