import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# признаки для картинки на плоскости
PLOT_FEATURES = ["worst radius", "worst texture"]


def load_cancer():
    data = load_breast_cancer()
    X = data.data
    # злокачественная = 1, доброкачественная = -1
    y = np.where(data.target == 0, 1.0, -1.0)
    return X, y, list(data.feature_names)


def prepare_data(test_size=0.25, seed=42, features=None):
    X, y, feature_names = load_cancer()

    # убираем остальные признаки
    if features is not None:
        idx = [feature_names.index(name) for name in features]
        X = X[:, idx]
        feature_names = list(features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, feature_names = prepare_data()
#     print("признаков:", len(feature_names))
#     print("train:", X_train.shape, "test:", X_test.shape)
#     print("доля класса +1: train", round((y_train == 1).mean(), 3), "test", round((y_test == 1).mean(), 3))
#     print("mean:", X_train.mean(axis=0).round(3).max(), "std:", X_train.std(axis=0).round(3).min())
#     X_train, X_test, y_train, y_test, feature_names = prepare_data(features=PLOT_FEATURES)
#     print("для картинки:", feature_names, X_train.shape, X_test.shape)
