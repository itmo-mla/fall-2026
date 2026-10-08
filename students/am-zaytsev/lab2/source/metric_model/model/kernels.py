import numpy as np


def rect(r):
    """Π(r) = 1[|r| ≤ 1]  — rectangular (uniform)."""
    r = np.asarray(r, dtype=float)
    return (np.abs(r) <= 1).astype(float)


def tri(r):
    """T(r) = (1 − |r|)·1[|r| ≤ 1]  — triangular."""
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= 1, 1.0 - np.abs(r), 0.0)


def epanechnikov(r):
    """E(r) = (1 − r²)·1[|r| ≤ 1]  — quadratic (Epanechnikov)."""
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= 1, 1.0 - r**2, 0.0)


def quartic(r):
    """Q(r) = (1 − r²)²·1[|r| ≤ 1]  — quartic (biweight)."""
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= 1, (1.0 - r**2) ** 2, 0.0)


def gauss(r):
    """G(r) = exp(−2 r²)  — Gaussian."""
    r = np.asarray(r, dtype=float)
    return np.exp(-2.0 * r**2)


# Convenient registry (handy for sweeps in your lab)
kernels_dict = {
    "rect": rect,
    "tri": tri,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "gauss": gauss,
}
