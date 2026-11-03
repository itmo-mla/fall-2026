from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from algorithms import (
    build_linear_classifier,
    calculate_threshold,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    select_parameters,
    solve_dual,
)
from metrics import accuracy_score, f1_score, precision_score, recall_score
from plots import plot_solutions
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

np.random.seed(30)

X, y = make_moons(n_samples=300, noise=0.2, random_state=30)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)

X_train_raw = X_train
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


def print_dual_result(name, C, lambdas, dual_value, y):
    print(f"Ядро: {name}, C = {C}, значение целевой функции: {dual_value:.6f}")
    print(f"Диапазон lambda: [{lambdas.min():.6f}, {lambdas.max():.6f}]")
    print(f"Остаток ограничения sum(lambda * y) = 0: {abs(np.dot(lambdas, y)):.2e}")


C_values = [0.1, 1.0, 10.0]
gamma_values = [0.1, 1.0, 10.0]

best_linear, cv_linear = select_parameters(
    X_train_raw, y_train, linear_kernel, C_values
)
best_poly, cv_poly = select_parameters(
    X_train_raw, y_train, polynomial_kernel, C_values
)
best_rbf, cv_rbf = select_parameters(
    X_train_raw,
    y_train,
    rbf_kernel,
    C_values,
    gamma_values,
)

cv_results = (
    [{"kernel": "linear", **row} for row in cv_linear]
    + [{"kernel": "polynomial", **row} for row in cv_poly]
    + [{"kernel": "rbf", **row} for row in cv_rbf]
)
print(pd.DataFrame(cv_results).to_string(index=False))

C_linear = best_linear["C"]
lambdas_linear, dual_linear = solve_dual(X_train, y_train, C_linear, linear_kernel)
print_dual_result("linear", C_linear, lambdas_linear, dual_linear, y_train)
w, w0 = build_linear_classifier(X_train, y_train, lambdas_linear, C_linear)
print(f"w = {w}, w0 = {w0:.6f}")

C_poly = best_poly["C"]
lambdas_poly, dual_poly = solve_dual(
    X_train,
    y_train,
    C_poly,
    polynomial_kernel,
)
print_dual_result("polynomial", C_poly, lambdas_poly, dual_poly, y_train)
w0_poly = calculate_threshold(X_train, y_train, lambdas_poly, C_poly, polynomial_kernel)

C_rbf = best_rbf["C"]
kernel_rbf = partial(rbf_kernel, gamma=best_rbf["gamma"])
lambdas_rbf, dual_rbf = solve_dual(X_train, y_train, C_rbf, kernel_rbf)
print_dual_result("rbf", C_rbf, lambdas_rbf, dual_rbf, y_train)
w0_rbf = calculate_threshold(X_train, y_train, lambdas_rbf, C_rbf, kernel_rbf)

models = [
    (f"Линейное, C={C_linear:g}", linear_kernel, lambdas_linear, w0, C_linear),
    (
        f"Полиномиальное, d=3, C={C_poly:g}",
        polynomial_kernel,
        lambdas_poly,
        w0_poly,
        C_poly,
    ),
    (
        f"RBF, C={C_rbf:g}, γ={best_rbf['gamma']:g}",
        kernel_rbf,
        lambdas_rbf,
        w0_rbf,
        C_rbf,
    ),
]
figures = Path(__file__).resolve().parent.parent / "figures"
figures.mkdir(exist_ok=True)
plot_solutions(X_train, y_train, models, figures / "svm_solutions.png")

reference_linear = SVC(C=C_linear, kernel="linear", tol=1e-8)
reference_linear.fit(X_train, y_train)

reference_poly = SVC(C=C_poly, kernel="poly", degree=3, gamma=1, coef0=1, tol=1e-8)
reference_poly.fit(X_train, y_train)

reference_rbf = SVC(C=C_rbf, kernel="rbf", gamma=best_rbf["gamma"], tol=1e-8)
reference_rbf.fit(X_train, y_train)

linear_scores = np.dot(X_test, w) - w0
poly_scores = np.dot(polynomial_kernel(X_test, X_train), lambdas_poly * y_train) - w0_poly
rbf_scores = np.dot(kernel_rbf(X_test, X_train), lambdas_rbf * y_train) - w0_rbf

results = []
comparisons = []
for name, scores, lambdas, reference in [
    ("linear", linear_scores, lambdas_linear, reference_linear),
    ("polynomial", poly_scores, lambdas_poly, reference_poly),
    ("rbf", rbf_scores, lambdas_rbf, reference_rbf),
]:
    predictions = np.where(scores >= 0, 1, -1)
    reference_predictions = reference.predict(X_test)
    for implementation, y_pred in [("own", predictions), ("sklearn SVC", reference_predictions)]:
        results.append(
            {
                "kernel": name,
                "model": implementation,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }
        )
    comparisons.append(
        {
            "kernel": name,
            "prediction_differences": np.sum(predictions != reference_predictions),
            "max_score_difference": np.max(np.abs(scores - reference.decision_function(X_test))),
            "support_own": np.sum(lambdas > 1e-6),
            "support_sklearn": reference.n_support_.sum(),
        }
    )

print(pd.DataFrame(results).to_string(index=False))
print(pd.DataFrame(comparisons).to_string(index=False))
