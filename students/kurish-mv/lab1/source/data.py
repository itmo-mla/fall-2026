from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "pima-indians-diabetes.csv"
COLUMNS = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
TARGET = "class"
ZERO_AS_MISSING = ["plas", "pres", "skin", "test", "mass"]


def load_pima(path=DATA_PATH):
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")  # на случай строки-заголовка
    return df.reset_index(drop=True)


def prepare_data(df, seed=42):
    X = df[COLUMNS[:-1]].copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    y = np.where(df[TARGET] == 1, 1.0, -1.0)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    # все статистики считаются только по train
    med = X_tr.median()
    X_tr, X_val, X_te = (a.fillna(med) for a in (X_tr, X_val, X_te))
    lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
    X_tr, X_val, X_te = (a.clip(lo, hi, axis=1) for a in (X_tr, X_val, X_te))
    mean, std = X_tr.mean(), X_tr.std(ddof=0)
    X_tr, X_val, X_te = (((a - mean) / std).to_numpy(float) for a in (X_tr, X_val, X_te))

    for a in (X_tr, X_val, X_te):
        assert np.isfinite(a).all(), "В данных остались NaN/inf"

    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val,
            "X_test": X_te, "y_test": y_te, "feature_names": COLUMNS[:-1]}
