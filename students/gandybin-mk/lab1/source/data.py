from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/"
    "data_banknote_authentication.txt"
)
FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]
COLUMNS = FEATURE_COLUMNS + ["class"]


def load_dataset():
    """Скачивает датасет из UCI и возвращает DataFrame с именами колонок."""
    with urlopen(DATASET_URL, timeout=30) as response:
        return pd.read_csv(response, header=None, names=COLUMNS)


def describe_dataset(df):
    """Возвращает размер, типы, пропуски, классы и числовую статистику."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing": df.isna().sum(),
        "class_counts": df["class"].value_counts(),
        "description": df.describe(),
    }


def prepare_data(df):
    """Выделяет признаки и преобразует метки 0/1 в -1/+1, сохраняя исходный df."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["class"].replace({0: -1, 1: 1}).to_numpy(dtype=float)
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Разделяет данные со стратификацией и стандартизирует по train."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
