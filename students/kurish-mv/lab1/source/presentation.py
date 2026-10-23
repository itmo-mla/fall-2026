import numpy as np

from margin import calculate_margin


def random_presentation(X, y, w, b, rng):
    return rng.permutation(len(X))


def margin_presentation(X, y, w, b, rng, eps=0.05):
    # чаще берём объекты с малым |M| (у границы): p ~ 1 / (|M| + eps)
    p = 1.0 / (np.abs(calculate_margin(X, y, w, b)) + eps)
    return rng.choice(len(X), size=len(X), p=p / p.sum())
