import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def add_bias(X):
    return np.hstack([-np.ones((X.shape[0], 1)), X])


def load_data(test_size=0.2, seed=42):
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    X_train = add_bias(scaler.transform(X_train))
    X_test = add_bias(scaler.transform(X_test))
    feature_names = ["bias"] + list(data.feature_names)
    return X_train, X_test, y_train, y_test, feature_names
