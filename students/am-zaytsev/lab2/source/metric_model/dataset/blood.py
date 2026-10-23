import os

import kagglehub
import pandas as pd

from .utils import draw_data_by_target, one_hot_encode

# Column holding the class label — adjust if your CSV uses a different name
LABEL_COL = "Class(Target)"
DATASET_DESC = """
Для выполнения лабораторной работы был выбран датасет
[донорство крови](https://www.kaggle.com/datasets/vstacknocopyright/blood-transfusion-service-center-data)\

Всего 2 класса. Сдавал ли человек кровь."""


def download_dataset() -> pd.DataFrame:
    # Download latest version
    data_path = kagglehub.dataset_download(
        "vstacknocopyright/blood-transfusion-service-center-data"
    )
    csv_name = next(f for f in os.listdir(data_path) if f.endswith(".csv"))
    df = pd.read_csv(os.path.join(data_path, csv_name))
    return df


def preprocess_data(df):
    # Some copies ship an extra "Id" column — drop it if present
    id_cols = [c for c in ("Id", "id") if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    X = df.drop(LABEL_COL, axis=1).to_numpy()
    Y = df[LABEL_COL].to_numpy()
    return X, one_hot_encode(Y)


def draw_data(df):
    return draw_data_by_target(df, LABEL_COL)
