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


def predict_parzen(X_train, y_train, X, k):
    if not 1 <= k < len(X_train):
        raise ValueError(
            "k must be between 1 and the number of training objects minus 1"
        )

    predictions = []

    for x in X:
        distances = calculate_distances(X_train, x)
        predictions.append(predict_parzen_from_distances(distances, y_train, k))

    return np.array(predictions)


def predict_parzen_from_distances(distances, y_train, k):
    classes = np.unique(y_train)
    # already k+1 (indexes from 0)
    h = np.partition(distances, k)[k]

    if h == 0:
        weights = (distances == 0).astype(float)
    else:
        weights = gaussian_kernel(distances / h)

    scores = np.array([np.sum(weights[y_train == cls]) for cls in classes])
    return choose_class(classes, scores, distances, y_train)


def choose_class(classes, scores, distances, y_train):
    winners = classes[scores == np.max(scores)]
    if len(winners) == 1:
        return winners[0]

    nearest_distances = [np.min(distances[y_train == cls]) for cls in winners]
    return winners[np.argmin(nearest_distances)]


def gaussian_kernel(r):
    return np.exp(-2 * r**2)


def loo(X, y, ks):
    ks = np.asarray(ks)
    if ks.size == 0 or np.any((ks < 1) | (ks > len(X) - 2)):
        raise ValueError("k must be between 1 and the number of objects minus 2")

    distances = np.array([calculate_distances(X, x) for x in X])
    np.fill_diagonal(distances, np.inf)

    risks = []
    for k in ks:
        predictions = np.array(
            [predict_parzen_from_distances(row, y, k) for row in distances]
        )
        risks.append(np.mean(predictions != y))

    return np.array(risks)
