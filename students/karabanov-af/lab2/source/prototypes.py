import numpy as np

import knn


def select(X, y, k=5, delta=0.0, max_errors=0):
    """STOLP: drop the noise, then grow the set of prototypes until it classifies the rest well.

    1. objects with a margin below delta are considered noise and are dropped;
    2. the object with the largest margin in every class becomes the first prototype;
    3. while there are too many errors, the worst classified object joins the prototypes.
    """
    D = knn.distances(X, X)
    np.fill_diagonal(D, np.inf)
    margins = knn.margins(D, y, y, k)
    clean = np.flatnonzero(margins > delta)
    prototypes = [clean[y[clean] == c][margins[clean[y[clean] == c]].argmax()] for c in range(y.max() + 1)]

    while True:
        current = knn.margins(D[np.ix_(clean, prototypes)], y[prototypes], y[clean], k)
        candidates = (current < 0) & ~np.isin(clean, prototypes)
        if candidates.sum() <= max_errors:
            return np.array(sorted(prototypes)), margins
        prototypes.append(clean[np.where(candidates, current, np.inf).argmin()])
