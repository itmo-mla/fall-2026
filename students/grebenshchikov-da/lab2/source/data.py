from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "uciml/glass/versions/1"
DATASET_FILE = "glass.csv"

FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
CLASS_NAMES = {
    1: "Оконное, флоат",
    2: "Оконное, обычное",
    3: "Автомобильное",
    5: "Тара",
    6: "Посуда",
    7: "Фары",
}


def load_data(test_size: float = 0.2, random_state: int = 42):
    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    data = pd.read_csv(dataset_dir / DATASET_FILE)
    data = data.drop(columns=["Id"], errors="ignore")

    X = data[FEATURES].to_numpy(dtype=float)
    source_labels = data["Type"].to_numpy(dtype=int)
    label_values = np.array(sorted(np.unique(source_labels)))
    label_to_index = {label: index for index, label in enumerate(label_values)}
    y = np.array([label_to_index[label] for label in source_labels], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Статистики считаются только по train, чтобы избежать утечки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_names = [CLASS_NAMES[label] for label in label_values]
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names
