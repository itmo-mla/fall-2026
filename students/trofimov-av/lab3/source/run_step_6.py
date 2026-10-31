import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC

from data import load_iris_split
from svm import KernelSVM


def compare_models(x_train, x_test, y_train, y_test, kernel):
    own = KernelSVM(c=1.0, kernel=kernel, gamma=0.25).fit(x_train, y_train)
    reference = SVC(C=1.0, kernel=kernel, gamma=0.25).fit(x_train, y_train)
    own_predictions = own.predict(x_test)
    reference_predictions = reference.predict(x_test)

    print(f"\nЯдро: {kernel}, C=1" + (", gamma=0.25" if kernel == "rbf" else ""))
    for name, predictions in (("Наш SVM", own_predictions), ("Sklearn SVC", reference_predictions)):
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", pos_label=1, zero_division=0
        )
        print(f"{name}: accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}")

    print(f"Совпадение предсказаний: {np.mean(own_predictions == reference_predictions):.0%}")
    print(f"Опорных векторов: наш SVM — {len(own.support)}, Sklearn — {len(reference.support_)}")


def main():
    x_train, x_test, y_train, y_test = load_iris_split()
    for kernel in ("linear", "rbf"):
        compare_models(x_train, x_test, y_train, y_test, kernel)


if __name__ == "__main__":
    main()
