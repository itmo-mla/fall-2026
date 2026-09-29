import numpy as np


def gaussian_kernel(values):
    values = np.asarray(values, dtype=float)
    return np.exp(-0.5 * values**2)


class ParzenKNN:
    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k должно быть положительным")
        self.k = k

    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("Матрица признаков должна быть непустой")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("Число меток должно совпадать с числом объектов")
        if self.k > len(features):
            raise ValueError("k не может быть больше числа обучающих объектов")
        self.features = features
        self.labels = labels
        self.classes = np.unique(labels)
        return self

    def _distances(self, features):
        differences = features[:, None, :] - self.features[None, :, :]
        return np.sqrt(np.sum(differences**2, axis=2))

    def class_scores(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.features.shape[1]:
            raise ValueError("Некорректная размерность признаков")

        distances = self._distances(features)
        bandwidths = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]
        bandwidths = np.maximum(bandwidths, np.finfo(float).eps)
        weights = gaussian_kernel(distances / bandwidths[:, None])
        scores = np.column_stack(
            [weights[:, self.labels == class_label].sum(axis=1) for class_label in self.classes]
        )
        return scores, bandwidths

    def predict(self, features):
        scores, _ = self.class_scores(features)
        return self.classes[np.argmax(scores, axis=1)]

    def predict_proba(self, features):
        scores, _ = self.class_scores(features)
        return scores / scores.sum(axis=1, keepdims=True)
