import numpy as np

from data import load_iris_split
from svm import solve_dual


def main():
    x_train, _, y_train, _ = load_iris_split()

    for kernel in ("linear", "rbf"):
        alpha = solve_dual(x_train, y_train, kernel=kernel, gamma=0.25)
        count = np.count_nonzero(alpha > 1e-6)
        print(f"{kernel}: задача решена, опорных векторов — {count}")


if __name__ == "__main__":
    main()
