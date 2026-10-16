from typing import List

import kagglehub
import pandas as pd

COLS = ['A_id', 'Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness',
        'Ripeness', 'Acidity', 'Quality']

DROP_COLS = ["A_id"]
TARGET = ["Quality"]


def download_data(url: str = "nelgiriyewithana/apple-quality") -> None:
    download_path = kagglehub.dataset_download(url, output_dir= "./data")
    print(f"Файл скачен в {download_path}")

def load_data(path: str ) -> pd.DataFrame:
    data = pd.read_csv(path)
    return  data

def prepare_data(data: pd.DataFrame, drop_cols: List[str], target) -> pd.DataFrame:
    if drop_cols:
        data = data.drop(columns= drop_cols)
    data = data.dropna()
    data[target] = data[target].replace({"good" : 1, "bad" : -1})
    return  data


