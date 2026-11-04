from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "alexandrepetit881234/fake-bills"
DATASET_FILE = "fake_bills.csv"
FEATURES = ["margin_up", "length"]


def load_data(sample_per_class: int = 150):
    path = kagglehub.dataset_download(DATASET, path=DATASET_FILE)
    data = pd.read_csv(Path(path), sep=";")
    data = data[[*FEATURES, "is_genuine"]].dropna()

    data["target"] = np.where(data["is_genuine"], 1, -1)

    # Ограничиваем размер двойственной задачи и выравниваем классы
    data = (
        data.groupby("target", group_keys=False)
        .sample(n=sample_per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Статистики считаются только по train, чтобы избежать утечки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X, y
