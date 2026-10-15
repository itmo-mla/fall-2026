import numpy as np

from data import load_iris_split
from svm import LinearSVM


def main():
    x_train, x_test, y_train, y_test = load_iris_split()
    model = LinearSVM(c=1.0).fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Опорных векторов: {len(model.support)}")
    print(f"Accuracy на test: {np.mean(predictions == y_test):.2%}")
    print(f"Ошибок: {np.count_nonzero(predictions != y_test)} из {len(y_test)}")


if __name__ == "__main__":
    main()
