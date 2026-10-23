import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]


def data_load():
    data = fetch_openml(data_id=1462, as_frame=True)
    X = data.data.copy()
    X.columns = FEATURE_NAMES
    y = np.where(data.target.to_numpy() == "1", 1, -1)
    return X, y


def split_and_scale(X, y, sub_size=250, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=sub_size, stratify=y_train, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X_train_sub, y_train_sub
