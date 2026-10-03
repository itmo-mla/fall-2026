# source/metric_model/model/SklearnKNN.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .AbstractKNN import KNN


class SklearnKNN(KNN):
    def __init__(self, k, weights="distance", *args):
        super().__init__()
        weights = weights.split(" ")[-1]
        self.k = k
        self.weights = weights
        self.metric = "minkowski"
        self.p = 2
        self.n_jobs = None
        self._model = KNeighborsClassifier(
            n_neighbors=self.k,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

    # Required by ABC. SklearnKNN delegates weighting to sklearn,
    # so this is only here to make the class concrete.
    def w(self, i):
        return np.int32(i < self.k)

    def train(self, X, y):
        # store on the base class (keeps self.X / self.y for the LOO harness)
        super().train(X, y)

        # sklearn wants 1-D integer labels, not one-hot
        if self.y.ndim > 1:
            y_labels = np.argmax(self.y, axis=1)
        else:
            y_labels = self.y

        self._model.fit(self.X, y_labels)
        return self

    def predict_prob(self, x):
        x = np.atleast_2d(x)
        proba = self._model.predict_proba(x)[0]

        # re-align sklearn's class order with the column order of self.y
        n_classes = self.y.shape[1] if self.y.ndim > 1 else len(np.unique(self.y))
        aligned = np.zeros(n_classes, dtype=np.float64)
        for col, cls in enumerate(self._model.classes_):
            aligned[int(cls)] = proba[col]
        return aligned
