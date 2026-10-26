import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


def generate_dataset(test_size=0.3, random_state=42):
    """
    Загружает Breast Cancer Wisconsin, стандартизует,
    делит на train/test и считает 2D-PCA для визуализации.

    Возвращает dict с ключами:
        X_train, X_test       — полные признаки (30D)
        y_train, y_test       — метки {-1, +1}
        X_train_2d, X_test_2d — проекция на 2 главные компоненты
        pca_ratio             — доля объяснённой дисперсии (2 числа)
        feature_names         — имена признаков
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)            # 0/1

    # Приводим метки к {-1, +1} для SVM
    y = np.where(y == 0, -1.0, 1.0)

    # Стандартизация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Стратифицированный split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y,
    )

    # PCA только для визуализации
    # Обучаем на train, применяем к test — так корректнее.
    pca = PCA(n_components=2, random_state=random_state)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    pca_ratio = pca.explained_variance_ratio_

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_2d": X_train_2d,
        "X_test_2d": X_test_2d,
        "pca_ratio": pca_ratio,
        "feature_names": list(data.feature_names),
    }


def save_dataset(dataset, train_path=TRAIN_PATH, test_path=TEST_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)

    feature_names = dataset["feature_names"]

    train_df = pd.DataFrame(dataset["X_train"], columns=feature_names)
    train_df["y"] = dataset["y_train"]
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame(dataset["X_test"], columns=feature_names)
    test_df["y"] = dataset["y_test"]
    test_df.to_csv(test_path, index=False)

    print(f"[data] сохранено: {train_path}")
    print(f"[data] сохранено: {test_path}")


def load_dataset(train_path=TRAIN_PATH, test_path=TEST_PATH, random_state=42):
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"train.csv / test.csv не найдены в {DATA_DIR}. "
            f"Сначала запусти generate_dataset() или main.py --regenerate."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_names = [c for c in train_df.columns if c != "y"]

    X_train = train_df[feature_names].to_numpy(dtype=np.float64)
    y_train = train_df["y"].to_numpy(dtype=np.float64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float64)
    y_test = test_df["y"].to_numpy(dtype=np.float64)

    pca = PCA(n_components=2, random_state=random_state)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_2d": X_train_2d,
        "X_test_2d": X_test_2d,
        "pca_ratio": pca.explained_variance_ratio_,
        "feature_names": feature_names,
    }