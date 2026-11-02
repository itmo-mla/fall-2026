
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Column holding the class label — adjust if your CSV uses a different name


def download_dataset(id=2) -> pd.DataFrame:
    return load_breast_cancer()


def preprocess_data(df):
    return df["data"]
