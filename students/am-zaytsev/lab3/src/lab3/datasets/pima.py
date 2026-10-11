import os

import pandas as pd

# Column holding the class label — adjust if your CSV uses a different name
LABEL_COL = "Outcome"


def download_dataset() -> pd.DataFrame:
    # Download latest version
    data_path = "~/.cache/kagglehub/datasets/jamaltariqcheema/pima-indians-diabetes-dataset/versions/1"
    df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))
    return df


def preprocess_data(df):
    # Some copies ship an extra "Id" column — drop it if present
    id_cols = [c for c in ("Id", "id") if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    X = df.drop(LABEL_COL, axis=1).to_numpy()
    Y = df[LABEL_COL].to_numpy()
    return X, Y * 2 - 1
