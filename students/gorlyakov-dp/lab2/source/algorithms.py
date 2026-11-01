import numpy as np


def calculate_distances(X, x):
    return np.linalg.norm(X - x, axis=1)


def predict_knn(X_train, y_train, X, k):
    if not 1 <= k <= len(X_train):
        raise ValueError("k must be between 1 and the number of training objects")

    predictions = []

    for x in X:
        distances = calculate_distances(X_train, x)
        nearest = np.argsort(distances)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        predictions.append(labels[np.argmax(counts)])

    return np.array(predictions)
