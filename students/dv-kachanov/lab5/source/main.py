import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logreg import autograd_gradient, gradient, irls, log_likelihood, newton_raphson, predict, predict_proba
from plots import plot_convergence, plot_probabilities


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# выбрать датасет для логистической регрессии;

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_test = np.column_stack((np.ones(len(X_test)), X_test))


# реализовать логистическую регрессию с помощью метода Ньютона-Рафсона и IRLS;

print("Отклонение автоград-градиента от замкнутой формулы:", np.max(np.abs(
    autograd_gradient(X_train, y_train, np.zeros(X_train.shape[1]))
    - gradient(X_train, y_train, np.zeros(X_train.shape[1]))
)))

w_newton, newton_history = newton_raphson(X_train, y_train)
w_irls, irls_history = irls(X_train, y_train)

print("Итераций Ньютона-Рафсона:", len(newton_history) - 1)
print("Итераций IRLS:", len(irls_history) - 1)
print("Функционал J(w) (Ньютон-Рафсон):", newton_history[-1])
print("Функционал J(w) (IRLS):", irls_history[-1])
print("Макс. отклонение весов Ньютон-Рафсон / IRLS:", np.max(np.abs(w_newton - w_irls)))

plot_convergence(newton_history, irls_history, OUTPUT_DIR / "convergence.png")


# показать эквивалентность с эталонной реализацией;

reference_model = LogisticRegression(C=1.0, solver="lbfgs", tol=1e-8, max_iter=20000)
reference_model.fit(X_train[:, 1:], y_train)

w_reference = np.concatenate(([reference_model.intercept_[0]], reference_model.coef_[0]))

print()
print("Макс. отклонение весов от эталона:", np.max(np.abs(w_newton - w_reference)))
print("Логарифмическое правдоподобие (эталон):", log_likelihood(X_train, y_train, w_reference))

own_proba = predict_proba(X_test, w_newton)
reference_proba = reference_model.predict_proba(X_test[:, 1:])[:, 1]

print("Макс. отклонение вероятностей от эталона:", np.max(np.abs(own_proba - reference_proba)))

plot_probabilities(own_proba, y_test, OUTPUT_DIR / "probabilities.png")


# оценить качество классификации;

results = [
    evaluate(y_test, predict(X_test, w_newton), "Newton-Raphson"),
    evaluate(y_test, predict(X_test, w_irls), "IRLS"),
    evaluate(y_test, reference_model.predict(X_test[:, 1:]), "sklearn LogisticRegression"),
]

results_df = pd.DataFrame(results)

print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))


# подготовить отчет.
