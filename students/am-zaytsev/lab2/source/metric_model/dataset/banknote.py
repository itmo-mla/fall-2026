import os

import kagglehub
import pandas as pd

from .utils import draw_data_by_target, one_hot_encode


def download_dataset() -> pd.DataFrame:
    # Download latest version
    data_path = kagglehub.dataset_download(
        "ritesaluja/bank-note-authentication-uci-data"
    )
    df = pd.read_csv(os.path.join(data_path, "BankNote_Authentication.csv"))
    return df


def preprocess_data(df):
    X = df.drop("class", axis=1).to_numpy()
    Y = df["class"].to_numpy()
    return X, one_hot_encode(Y)


def draw_data(df):
    return draw_data_by_target(df, "class")
