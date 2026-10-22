import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from lab5.dataset.utils import load_dataset
from lab5.model import LogReg


def print_confusion_matrix(cm) -> None:
    width = max(len(str(int(value))) for row in cm for value in row)
    for row in cm:
        cells = " ".join(f"{int(value):>{width}}" for value in row)
        print(f"  {cells}")


def print_result(name: str, accuracy: float, cm) -> None:
    print(name)
    print(f"  accuracy: {accuracy:.4f}")
    print("  confusion matrix:")
    print_confusion_matrix(cm)
    print()


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model_sk = LogisticRegression(
        solver="newton-cholesky",  # closest to Newton/IRLS
        C=np.inf,  # no L2 regularization
        tol=1e-4,
        max_iter=10_000,
    )
    model_sk.fit(X_train, y_train[:, 0])

    acc_sk = model_sk.score(X_test, y_test[:, 0])
    cm_sk = confusion_matrix(model_sk.predict(X_test), y_test)
    print_result("sklearn LogisticRegression", acc_sk, cm_sk)

    model_my = LogReg(max_iteration=10_000, h=1, tol=1e-4)
    model_my.train(X_train, y_train)
    acc_my = (model_my.eval(X_test) == y_test).mean()
    cm_my = confusion_matrix(model_my.eval(X_test), y_test)
    print_result("custom LogReg (Newton-Raphson / IRLS)", acc_my, cm_my)

    results_compare_acc = (
        model_sk.predict(X_test) == model_my.eval(X_test)[:, 0]
    ).mean()
    print(f"Models prediction match: {results_compare_acc:.2%}")


if __name__ == "__main__":
    main()
