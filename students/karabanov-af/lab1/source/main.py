import os

import numpy as np

import plots
from data import add_bias, load_binary_iris, standardize

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")

def margins(w, X, y):
    return y * (X @ w)


def main():
    X, y = load_binary_iris()
    X = add_bias(standardize(X))
    print(f"объектов: {X.shape[0]}, признаков с bias: {X.shape[1]}")

    rng = np.random.default_rng(0)
    d = X.shape[1]
    w = rng.uniform(-1 / (2 * d), 1 / (2 * d), size=d)
    print("случайные веса:", np.round(w, 3))

    m = margins(w, X, y)
    print(f"отступы: min {m.min():.3f}, max {m.max():.3f}, ошибок (M < 0): {(m < 0).sum()}")

    plots.plot_sorted_margins(m, "Отступы объектов при случайных весах",
                              os.path.join(IMAGES, "margins_random.png"))


if __name__ == "__main__":
    main()
