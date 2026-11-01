import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MISSING_AS_ZERO = ["plas", "pres", "skin", "insu", "mass"]


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def load_data(test_size=0.2, seed=42):
    data = fetch_openml("diabetes", version=1, as_frame=True)
    X = data.data.astype(float)
    y = np.where(data.target == "tested_positive", 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_test = X_train.copy(), X_test.copy()
    for column in MISSING_AS_ZERO:
        median = X_train.loc[X_train[column] > 0, column].median()
        X_train.loc[X_train[column] == 0, column] = median
        X_test.loc[X_test[column] == 0, column] = median

    scaler = StandardScaler().fit(X_train)
    X_train = add_bias(scaler.transform(X_train))
    X_test = add_bias(scaler.transform(X_test))
    feature_names = ["bias"] + list(data.feature_names)
    return X_train, X_test, y_train, y_test, feature_names
