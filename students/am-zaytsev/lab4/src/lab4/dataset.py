import os

import numpy as np
import pandas as pd

# Column holding the class label — adjust if your CSV uses a different name


def find(root):
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        if os.path.isdir(fpath):
            yield from find(fpath)
            continue
        if str(fname).endswith(".csv"):
            yield fpath


def download_dataset(id=2) -> pd.DataFrame:
    # Download latest version
    data_root = "/home/artem/.cache/kagglehub/datasets/"
    data_path = list(find(data_root))[id]
    df = pd.read_csv(data_path)
    if 'Id' in df:
        del df['Id']
    df = df.drop(columns=list(df)[-1])
    return df


def preprocess_data(df):
    X = np.float32(df.to_numpy())
    return X


if __name__ == "__main__":
    # print(find("/home/artem/.cache/kagglehub/datasets/"))
    for i in find():
        print(i)
