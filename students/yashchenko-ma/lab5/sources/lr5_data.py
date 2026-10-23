"""Данные для ЛР5: сбалансированная подвыборка (get_data из data.py без изменений)
и полная несбалансированная выборка (10000 строк, 3.4% отказов)."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import get_data, load_raw, FEATURES, RANDOM_STATE

FEATURE_NAMES = FEATURES + ["type_L", "type_M"]
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def save_csv(df, name):
    """Сохранить таблицу в results/<name> (UTF-8, разделитель запятая, числа без округления)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"[csv] сохранено: {path}")


def get_subsample_split():
    """Подвыборка 1:2 (все отказы + 2x случайных без отказа), метки {-1,+1}."""
    return get_data()


def get_full_split(test_size=0.25, random_state=RANDOM_STATE):
    """Полная выборка, стратифицированный split, StandardScaler обучен только на train."""
    df = load_raw()
    dummies = pd.get_dummies(df["Type"], prefix="type", drop_first=True).astype(float)
    X = pd.concat([df[FEATURES], dummies], axis=1).values.astype(float)
    y = np.where(df["target"].values == 1, 1, -1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te
