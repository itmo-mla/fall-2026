from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def fit_reference(X_train, y_train, C=1.0, gamma=1.0, kernel="rbf"):
    """
        Эталонный SVM на sklearn.svm.SVC.
        Для линейного ядра gamma не передаётся — иначе sklearn
        выдаёт UserWarning: 'gamma parameter is irrelevant for linear kernel'.
    """
    if kernel == "linear":
        clf = SVC(kernel="linear", C=C)
    else:
        clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    return clf


def evaluate(model, X, y):
    pred = model.predict(X)
    return accuracy_score(y, pred), pred