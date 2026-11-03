import kagglehub
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE



COLS = ['A_id', 'Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness',
        'Ripeness', 'Acidity', 'Quality']

DROP_COLS = ["A_id"]
TARGET = ["Quality"]

def download_data(path = "nelgiriyewithana/apple-quality") -> None :
    download_path = kagglehub.dataset_download(path, output_dir="./data")
    print(f"Файлы скачены в: {download_path}")




def load_data(path = "./data/apple_quality.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def print_data_overview(data : pd.DataFrame) -> None:
    print(data.head(3))

    print(data.info())

    print(data.describe())

    print(data.isna().sum())



def prepare_data(data: pd.DataFrame, drop_cols=None):
    if drop_cols is None:
        drop_cols = DROP_COLS

    data = data.dropna()
    data = data.drop(columns = drop_cols)

    data[TARGET] = data[TARGET].replace({"good": 1.0, "bad": -1.0})


    return data



