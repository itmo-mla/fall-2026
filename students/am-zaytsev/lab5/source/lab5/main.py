from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from lab5.dataset.utils import load_dataset
from lab5.model import LogReg


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model_sk = LogisticRegression(max_iter=10_000)
    model_sk.fit(X_train, y_train[:, 0])

    acc_sk = model_sk.score(X_test, y_test[:, 0])
    cm_sk = confusion_matrix(model_sk.predict(X_test), y_test)

    model_my = LogReg(max_iteration=10_000, h=1, tol=1e-4)
    model_my.train(X_train, y_train)
    acc_my = (model_my.eval(X_test) == y_test).mean()
    cm_my = confusion_matrix(model_my.eval(X_test), y_test)

    results_compare_acc = (
        model_sk.predict(X_test) == model_my.eval(X_test)[:, 0]
    ).mean()


if __name__ == "__main__":
    main()
