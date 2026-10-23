import kagglehub
import pandas as pd
import os
# Column holding the class label — adjust if your CSV uses a different name
LABEL_COL = "Outcome"
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"


def download_dataset() -> pd.DataFrame:
    print("[data] Loading dataset")
    try:
        data_path = kagglehub.dataset_download(
            "jamaltariqcheema/pima-indians-diabetes-dataset"
        )
    except:
        data_path = os.path.expanduser('~/.cache/kagglehub/datasets/jamaltariqcheema/pima-indians-diabetes-dataset/versions/1/diabetes.csv')
    df = pd.read_csv(data_path)
    print("[data] Dataset loaded")
    return df


def preprocess_data(df):
    # Some copies ship an extra "Id" column — drop it if present
    id_cols = [c for c in ("Id", "id") if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    X = df.drop(LABEL_COL, axis=1).to_numpy()
    Y = df[LABEL_COL].to_numpy()
    return X, Y * 2 - 1
