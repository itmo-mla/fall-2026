import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from plots import plot_convergence, plot_probabilities

from logreg import (
    autograd_gradient,
    correct_class_proba,
    decision,
    gradient,
    irls,
    newton_raphson,
    objective_Q,
    predict01,
    to_signed_labels,
)


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
y01 = dataset.target

F_train, F_test, y01_train, y01_test = train_test_split(
    X,
    y01,
    test_size=0.25,
    random_state=42,
    stratify=y01
)

# Метки датасета {0, 1} -> нотация лекции {-1, +1} (слайд 20).
y_tilde_train = to_signed_labels(y01_train)
y_tilde_test = to_signed_labels(y01_test)

# Матрица данных лекции F = (f_j(x_i)) размера l x n.
# Свободный член добавлен константным признаком f_0(x_i) = 1 (первый столбец).
F_train = np.hstack([np.ones((F_train.shape[0], 1)), F_train])
F_test = np.hstack([np.ones((F_test.shape[0], 1)), F_test])

print("Размер F_train:", F_train.shape)
print("Размер F_test:", F_test.shape)


# реализовать логистическую регрессию с помощью метода Ньютона-Рафсона и IRLS;

print("Отклонение автоград-градиента от замкнутой формулы:", np.max(np.abs(
    autograd_gradient(F_train, y_tilde_train, np.zeros(F_train.shape[1]))
    - gradient(F_train, y_tilde_train, np.zeros(F_train.shape[1]))
)))

w_newton, newton_history, newton_iters = newton_raphson(F_train, y_tilde_train, h_t=1.0)
w_irls, irls_history, irls_iters = irls(F_train, y_tilde_train, h_t=1.0)

print("Шаг h_t (оба метода):", 1.0)
print("Итераций Ньютона-Рафсона:", newton_iters)
print("Итераций IRLS:", irls_iters)
print("Функционал Q(w) (Ньютон-Рафсон):", newton_history[-1])
print("Функционал Q(w) (IRLS):", irls_history[-1])
print("Макс. отклонение весов Ньютон-Рафсон / IRLS:", np.max(np.abs(w_newton - w_irls)))

plot_convergence(newton_history, irls_history, OUTPUT_DIR / "convergence.png")


# показать сопоставление с эталонной реализацией;

# Эталон получает ту же матрицу F (с константным признаком) и без регуляризации,
# поэтому постановки полностью совпадают.
reference_model = LogisticRegression(
    penalty=None,
    fit_intercept=False,
    solver="lbfgs",
    tol=1e-8,
    max_iter=20000,
)
with warnings.catch_warnings():
    # Выборка линейно разделима, поэтому оценка максимального правдоподобия
    # без регуляризации не имеет конечного решения: lbfgs останавливается
    # по лимиту вычислений.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    reference_model.fit(F_train, y01_train)

print("Итераций sklearn (lbfgs):", int(reference_model.n_iter_[0]))

reference_pred = reference_model.predict(F_test)
own_pred = predict01(F_test, w_newton)
own_pred_irls = predict01(F_test, w_irls)

print()
print("Доля совпадающих предсказаний (наш Ньютон vs sklearn):",
      float(np.mean(own_pred == reference_pred)))
print("Доля совпадающих предсказаний (наш IRLS vs sklearn):",
      float(np.mean(own_pred_irls == reference_pred)))

test_sigma = correct_class_proba(F_test, y_tilde_test, w_newton)

plot_probabilities(test_sigma, y_tilde_test, OUTPUT_DIR / "probabilities.png")


# оценить качество классификации;

results = [
    evaluate(y01_test, own_pred, "Newton-Raphson"),
    evaluate(y01_test, own_pred_irls, "IRLS"),
    evaluate(y01_test, reference_pred, "sklearn LogisticRegression"),
]

results_df = pd.DataFrame(results)

print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))
print()
print("Q(w) на обучающей выборке:", objective_Q(F_train, y_tilde_train, w_newton))
print("Q(w) на тестовой выборке:", objective_Q(F_test, y_tilde_test, w_newton))
print("Доля верных знаков (train):", float(np.mean(decision(F_train, w_newton) == y_tilde_train)))
print("Доля верных знаков (test):", float(np.mean(decision(F_test, w_newton) == y_tilde_test)))


# подготовить отчет.
