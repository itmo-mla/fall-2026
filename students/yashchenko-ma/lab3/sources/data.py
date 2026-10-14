"""Шаги 1-2: загрузка AI4I 2020, предобработка, сбалансированная подвыборка, train/test, стандартизация."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear']
RANDOM_STATE = 42
OPENML_DATA_ID = 42890

def load_raw():
    """
    Порядок попыток: локальный CSV (offline) -> OpenML (нужен интернет
    только при первом запуске на машине)."""
    local_path = "ai4i2020.csv"   # Прямой путь к файлу    

    if local_path and os.path.isfile(local_path):
        print(f"[data.py] читаю локальный файл: {local_path} (offline)")
        df = pd.read_csv(local_path)
    else:
        print(f"[data.py] файл не найден - пробую sklearn.datasets.fetch_openml(data_id="
              f"{OPENML_DATA_ID}) (нужен интернет при первом запуске)")
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
        df = data.frame

    # Удаляем возможные пробелы вокруг названий колонок
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rot_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",

        "Air temperature": "air_temp",
        "Process temperature": "process_temp",
        "Rotational speed": "rot_speed",
        "Torque": "torque",
        "Tool wear": "tool_wear",

        "Machine failure": "target",
        "Target": "target",
    })

    required_columns = FEATURES + ["Type", "target"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"В CSV отсутствуют колонки: {missing_columns}\n"
            f"Фактические колонки: {list(df.columns)}"
        )

    print(df.shape)
    print(df['target'].value_counts())
    print(f"Доля отказов: {df['target'].mean():.4f}")

    return df


def make_subsample(df, seed=42):
    """Все отказы + случайные объекты без отказа (1:2). Метки {-1, +1}."""
    type_dummies = pd.get_dummies(df['Type'], prefix='type', drop_first=True)
    X_full = pd.concat([df[FEATURES], type_dummies], axis=1)
    y_full = df['target'].values

    rng = np.random.RandomState(seed)
    idx_pos = df.index[df['target'] == 1].to_numpy()
    idx_neg = rng.choice(df.index[df['target'] == 0].to_numpy(), size=len(idx_pos) * 2, replace=False)
    idx_sample = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(idx_sample)

    X = X_full.loc[idx_sample].values
    y = np.where(y_full[idx_sample] == 1, 1, -1)
    return X, y


def load_and_split(test_size=0.25, random_state=RANDOM_STATE):
    """Загружает датасет, делит на train/test (со стратификацией по классам)
    и масштабирует признаки (StandardScaler, обучен только на train)."""
    df = load_raw()

    X, y = make_subsample(df, random_state)
    print(f"Размер подвыборки: {X.shape}, положительных: {(y == 1).sum()}, отрицательных: {(y == -1).sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler().fit(X_train) # StandardScaler - метрические методы (KNN, Парзен) считают евклидово расстояние $\rho(x,x_i)=\sqrt{\sum_j (x^j-x_i^j)^2}$
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def get_data(random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = load_and_split(
        random_state=random_state
    )
    print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
