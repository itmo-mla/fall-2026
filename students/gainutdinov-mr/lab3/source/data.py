"""Загрузка и подготовка данных для лабораторной работы № 3."""

# TODO: выбрать датасет, реализовать загрузку, предобработку и разбиение выборки.
# Датасет - teejmahal20/airline-passenger-satisfaction
# Его предобработка взята из лаб1
# Осталось только переоформить в класс
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


def prepare_data(data_path, DROP_NA=True):

    df = pd.read_csv(data_path)
    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)
    if DROP_NA:
        df.dropna(inplace=True)

    rule_satisfied = {
        "satisfied": 1,
        "neutral or dissatisfied": -1,
    }  # 1 and -1 потому что у нас бинарная классификация
    rule_gender = {"Male": 0, "Female": 1}
    rule_customer_type = {"Loyal Customer": 0, "disloyal Customer": 1}
    rule_type_of_travel = {"Personal Travel": 0, "Business travel": 1}
    rule_class = {"Eco": 0, "Eco Plus": 1, "Business": 2}  # Так как их >2, пропишу явно

    df.replace({"satisfaction": rule_satisfied}, inplace=True)
    df.replace({"Gender": rule_gender}, inplace=True)
    df.replace({"Customer Type": rule_customer_type}, inplace=True)
    df.replace({"Type of Travel": rule_type_of_travel}, inplace=True)
    df.replace({"Class": rule_class}, inplace=True)

    return df


def standardize_data(X_train, X_test):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Чтобы не делить на ноль, где например постоянные признаки.
    train_std = np.where(train_std == 0, 1.0, train_std)

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    return X_train_scaled, X_test_scaled, train_mean, train_std


if __name__ == "__main__":
    path = Path(
        kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")
    )
    train_path = path / "train.csv"
    test_path = path / "test.csv"

    train_data = prepare_data(train_path)
    test_data = prepare_data(test_path)
    X_train = train_data.drop(["satisfaction"], axis=1)
    y_train = train_data["satisfaction"]
    X_test = test_data.drop(["satisfaction"], axis=1)
    y_test = test_data["satisfaction"]
    X_train = X_train.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)

    X_train, X_test, feature_mean, feature_std = standardize_data(
        X_train,
        X_test,
    )
    # Первый столбец равен единице: его коэффициент и есть свободный член w0.
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
