import numpy as np
import pandas as pd
from algorithms import (
    choose_by_margin,
    choose_random,
    calculate_margin,
    initialize_by_correlation,
    initialize_random,
    train_classifier,
    train_steepest,
)
from metrics import evaluate_model, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

np.random.seed(30)

data = load_breast_cancer()

X = data.data
y = np.where(data.target == 0, -1, 1)
X = np.column_stack((X, np.ones(len(X))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)

results = []

w0_corr = initialize_by_correlation(X_train, y_train)
w, Q, Q_history = train_classifier(X_train, y_train, w0_corr, choose_random)
results.append(evaluate_model("correlation + SGD", X_test, y_test, w))

w_multistart = None
old_loss = np.inf
for _ in range(10):
    w0 = initialize_random(X_train)
    w, Q, Q_history = train_classifier(X_train, y_train, w0, choose_random)
    M_train = calculate_margin(X_train, y_train, w)
    train_loss = np.mean((1 - M_train) ** 2)

    if train_loss < old_loss:
        old_loss = train_loss
        w_multistart = w

results.append(evaluate_model("multistart + SGD", X_test, y_test, w_multistart))

w0_random = initialize_random(X_train)

np.random.seed(30)
w_random, Q, Q_history = train_classifier(
    X_train, y_train, w0_random.copy(), choose_random
)
results.append(evaluate_model("random init + random", X_test, y_test, w_random))

np.random.seed(30)
w_margin, Q, Q_history = train_classifier(
    X_train, y_train, w0_random.copy(), choose_by_margin
)
results.append(evaluate_model("random init + margin", X_test, y_test, w_margin))

np.random.seed(31)
w_steepest, Q, Q_history = train_steepest(
    X_train, y_train, w0_corr, choose_random
)
results.append(evaluate_model("steepest descent", X_test, y_test, w_steepest))

ridge = RidgeClassifier(fit_intercept=False)
with np.errstate(all="ignore"):
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
results.append(
    {
        "model": "RidgeClassifier",
        "accuracy": accuracy_score(y_test, y_pred_ridge),
        "precision": precision_score(y_test, y_pred_ridge),
        "recall": recall_score(y_test, y_pred_ridge),
        "f1": f1_score(y_test, y_pred_ridge),
    }
)

print(pd.DataFrame(results).to_string(index=False))
