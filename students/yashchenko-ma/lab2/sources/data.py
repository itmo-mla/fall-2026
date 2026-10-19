import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OPENML_DATA_ID = 44972          # "red_wine" на OpenML, см. README.md


def load_wine_quality():
    """Возвращает (X, y): X -- 11 числовых признаков, y -- бинарная метка
    (1, если quality >= 7 -- "хорошее вино", иначе 0).

    Порядок попыток: локальный CSV (offline) -> OpenML (нужен интернет
    только при первом запуске на машине)."""
    local_path = "winequality-red.csv"   # Прямой путь к файлу
    
    if local_path and os.path.isfile(local_path):
        print(f"[data.py] читаю локальный файл: {local_path} (offline)")
        # df = pd.read_csv(local_path)
        df = pd.read_csv("winequality-red.csv")
        target_col = "quality"
    else:
        print(f"[data.py] файл не найден - пробую sklearn.datasets.fetch_openml(data_id="
              f"{OPENML_DATA_ID}) (нужен интернет при первом запуске)")
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
        df = data.frame
        target_col = "quality" if "quality" in df.columns else data.target_names[0]

    y = (df[target_col].astype(int) >= 7).astype(int).values
    X = df.drop(columns=[target_col]).values.astype(float)
    return X, y


def load_and_split(test_size=0.2, random_state=RANDOM_STATE):
    """Загружает датасет, делит на train/test (со стратификацией по классам)
    и масштабирует признаки (StandardScaler, обучен только на train)."""
    X, y = load_wine_quality()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler().fit(X_train) # StandardScaler - метрические методы (KNN, Парзен) считают евклидово расстояние $\rho(x,x_i)=\sqrt{\sum_j (x^j-x_i^j)^2}$
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"train: {X_train.shape}, баланс классов {np.bincount(y_train)}")
    print(f"test:  {X_test.shape}, баланс классов {np.bincount(y_test)}")
