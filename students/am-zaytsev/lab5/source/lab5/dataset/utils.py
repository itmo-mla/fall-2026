import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw():
    return load_breast_cancer()


def preprocess_raw(raw_data):
    X, y = raw_data.data, raw_data.target
    scaler = StandardScaler()
    y = y[..., np.newaxis]
    y = y * 2 - 1
    # Split and train instantly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_dataset():
    raw_data = load_raw()
    return preprocess_raw(raw_data)
