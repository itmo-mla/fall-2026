import numpy as np
from scipy.spatial.distance import cdist


def linear(X1, X2):
    return X1 @ X2.T


def polynomial(X1, X2, degree=3, coef0=1.0):
    return (X1 @ X2.T + coef0) ** degree


def rbf(X1, X2, gamma=0.5):
    return np.exp(-gamma * cdist(X1, X2, metric="sqeuclidean"))
