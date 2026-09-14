import numpy as np


def linear_ab(y):

    x = np.asarray(range(len(y)), dtype=float)
    y = np.asarray(y, dtype=float)

    n = x.size
    if n != y.size:
        raise ValueError("x and y must have the same length")

    mean_x = x.mean()
    mean_y = y.mean()

    # covariance and variance (population versions, denominator n)
    cov_xy = np.sum((x - mean_x) * (y - mean_y)) / n
    var_x = np.sum((x - mean_x) ** 2) / n

    if var_x == 0:
        raise ValueError("x has zero variance; a is undefined")

    a = cov_xy / var_x
    b = mean_y - a * mean_x

    return a, b
