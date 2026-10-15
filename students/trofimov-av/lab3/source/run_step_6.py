import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC

from data import load_iris_split
from svm import LinearSVM


def main():
    x_train, x_test, y_train, y_test = load_iris_split()
    own = LinearSVM(c=1.0).fit(x_train, y_train)
    reference = SVC(C=1.0, kernel="linear").fit(x_train, y_train)
    own_predictions = own.predict(x_test)
    reference_predictions = reference.predict(x_test)

    for name, predictions in (("Наш SVM", own_predictions), ("Sklearn SVC", reference_predictions)):
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", pos_label=1, zero_division=0
        )
        print(f"{name}: accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}")

    print(f"Совпадение предсказаний: {np.mean(own_predictions == reference_predictions):.0%}")
    print(f"Опорных векторов: наш SVM — {len(own.support)}, Sklearn — {len(reference.support_)}")


if __name__ == "__main__":
    main()
