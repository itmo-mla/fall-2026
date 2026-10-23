import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def add_bias(X):
    return np.hstack([-np.ones((X.shape[0], 1)), X])


def load_data(test_size=0.2, val_size=0.2, seed=42):
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 1, 1, -1)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    X_train = add_bias(scaler.transform(X_train))
    X_val = add_bias(scaler.transform(X_val))
    X_test = add_bias(scaler.transform(X_test))
    feature_names = ["bias"] + list(data.feature_names)
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
