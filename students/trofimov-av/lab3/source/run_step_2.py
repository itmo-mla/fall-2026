import numpy as np

from data import load_iris_split
from svm import solve_dual


def main():
    x_train, x_test, y_train, _ = load_iris_split()
    alpha = solve_dual(x_train, y_train)

    print(f"Iris: train={len(x_train)}, test={len(x_test)}, признаков={x_train.shape[1]}")
    print("Двойственная задача решена, C=1")
    print(f"Опорных векторов: {np.count_nonzero(alpha > 1e-6)}")


if __name__ == "__main__":
    main()
