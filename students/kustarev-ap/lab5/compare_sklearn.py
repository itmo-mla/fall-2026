import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

from logreg_irls import add_bias, sigmoid


def fit_sklearn(X_train, y_train, lam):
    model = LogisticRegression(penalty="l2", C=1 / lam, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compare(w_mine, model, X_test, y_test):
    p_mine = sigmoid(add_bias(X_test) @ w_mine)
    p_skl = model.predict_proba(X_test)[:, 1]

    w_skl = np.concatenate([model.intercept_, model.coef_[0]])
    y_pred_mine = (p_mine >= 0.5).astype(int)
    y_pred_skl = (p_skl >= 0.5).astype(int)

    return {
        "w_skl": w_skl,
        "coef_diff": np.max(np.abs(w_mine - w_skl)),
        "prob_diff": np.max(np.abs(p_mine - p_skl)),
        "acc_mine": accuracy_score(y_test, y_pred_mine),
        "acc_skl": accuracy_score(y_test, y_pred_skl),
        "logloss_mine": log_loss(y_test, p_mine),
        "logloss_skl": log_loss(y_test, p_skl),
        "cm_mine": confusion_matrix(y_test, y_pred_mine),
        "cm_skl": confusion_matrix(y_test, y_pred_skl),
    }
