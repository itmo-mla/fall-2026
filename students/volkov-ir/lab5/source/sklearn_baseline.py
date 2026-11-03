import numpy as np
from sklearn.linear_model import LogisticRegression


def calculate_sklearn_logreg(X_train, y_train, X_test, solver):
    # C = inf - без регуляризации, смещение уже в столбце из единиц
    model = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver, tol=1e-12, max_iter=1000).fit(X_train, y_train)

    # 2 столбца: вероятность класса -1 и класса +1, берем второй
    proba = model.predict_proba(X_test)[:, 1]
    return model.coef_[0], proba, model.predict(X_test), model.n_iter_[0]


def calculate_comparison(X_test, w, w_sk, proba_sk, pred_sk):
    z = X_test @ w
    proba = 1 / (1 + np.exp(-z))
    pred = np.sign(z)

    # наибольшее расхождение с sklearn
    diff_w = np.max(np.abs(w - w_sk))
    diff_proba = np.max(np.abs(proba - proba_sk))

    # сколько предсказаний не совпало
    count_pred = np.sum(pred != pred_sk)

    return diff_w, diff_proba, count_pred
