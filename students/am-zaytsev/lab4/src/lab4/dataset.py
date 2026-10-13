import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

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
    return load_breast_cancer()


def preprocess_data(df):
    return df["data"]


if __name__ == "__main__":
    # print(find("/home/artem/.cache/kagglehub/datasets/"))
    for i in find():
        print(i)
