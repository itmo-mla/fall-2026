import time
import numpy as np
from sklearn.svm import SVC


def calculate_sklearn_svm(X_train, y_train, X_test, kernel, d, C, gamma):
    # coef0 = 1, чтобы poly совпало с нашим (gamma * <x, x'> + 1)^d
    model = SVC(C=C, kernel=kernel, degree=d, gamma=gamma if kernel != "linear" else "scale", coef0=1)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    # dual_coef_ хранит lam * y только для опорных, собираем полный вектор lam
    lam = np.zeros(len(y_train))
    lam[model.support_] = np.abs(model.dual_coef_[0])

    # у sklearn знак свободного члена обратный
    w0 = -model.intercept_[0]
    return lam, w0, model.predict(X_test), end - start


def calculate_comparison(lam, lam_sk, w0, w0_sk, pred, pred_sk):
    # наибольшее расхождение с sklearn
    diff_lam = np.max(np.abs(lam - lam_sk))
    diff_w0 = abs(w0 - w0_sk)

    # сколько предсказаний не совпало
    count_pred = np.sum(pred != pred_sk)

    return diff_lam, diff_w0, count_pred
