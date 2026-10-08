import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from metric_model.visual.utils import draw_dataset


def one_hot_encode(row: np.array):
    row_2d = row.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform(row_2d)
    return one_hot


def remove_row(df, row_i):
    mask = np.ones(df.shape[0])
    mask[row_i] = 0
    mask = mask == 1
    return df[mask]


def draw_data_by_target(df: pd.DataFrame, target_name: str):
    # Features and labels
    features = [i for i in list(df) if i != target_name]
    X = df[features].values
    y = np.float32(df[target_name].values)
    y -= y.min()
    y /= y.max()
    y = y[:, None]
    draw_dataset(X, y)
