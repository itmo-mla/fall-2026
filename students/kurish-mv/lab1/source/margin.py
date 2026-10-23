import numpy as np


def calculate_margin(X, y, w, b=0.0):
    return y * (X @ w + b)
