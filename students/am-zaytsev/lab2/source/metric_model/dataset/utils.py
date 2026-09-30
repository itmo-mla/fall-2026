import os

import kagglehub
import pandas as pd


def download_dataset() -> pd.DataFrame:
    # Download latest version
    data_path = kagglehub.dataset_download(
        "ritesaluja/bank-note-authentication-uci-data"
    )
    df = pd.read_csv(os.path.join(data_path, "BankNote_Authentication.csv"))
    return df
