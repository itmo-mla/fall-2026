from pathlib import Path

import report
from data_preprocessing import prepare_data
from model import calculate_loo, calculate_best_k, calculate_predict, calculate_margin, calculate_start_omega, greedy_addition
from report import (evaluate, find_errors, print_loo, print_metrics, print_sklearn_check, print_prototypes,
                    plot_loo_risk, plot_prototype_map, plot_ccv_history)
from sklearn_baseline import run_sklearn_baseline

report.SAVE_DIR = Path(__file__).parent.parent / "images"

X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data()
n_classes = len(class_names)


# k_values – какие k перебираются в LOO, окну Парзена нужен (k+1)-й сосед среди l-1 объектов, поэтому k не больше l-2
k_values = list(range(1, len(X_train) - 1))
methods = ["knn", "parzen", "parzen_lecture"]

SHOW_PLOTS = True


# 3, 4 – подбор k по LOO на train
all_loo = {}
best_k = {}
for method in methods:
    all_loo[method] = calculate_loo(X_train, y_train, k_values, method)
    best_k[method] = calculate_best_k(k_values, all_loo[method])
print_loo(all_loo, best_k, k_values, len(X_train))


# 5 – качество на test с лучшим k
test_rows = []
for method in methods:
    pred_labels = calculate_predict(X_train, y_train, X_test, best_k[method], method)
    test_rows.append(("своя", report.SHORT_NAMES[method], best_k[method], evaluate(y_test, pred_labels, n_classes)))

# 5 – эталон sklearn
sklearn_results = run_sklearn_baseline(X_train, y_train, X_test)
print_sklearn_check(all_loo, sklearn_results)
for name in sklearn_results:
    result = sklearn_results[name]
    test_rows.append(("sklearn", report.SHORT_NAMES[name], result["k"], evaluate(y_test, result["pred"], n_classes)))
print_metrics("Качество на test", test_rows, len(X_test))


# 6 – отбор эталонов, старт по отступу окна Парзена с его лучшим k
margins = calculate_margin(X_train, y_train, best_k["parzen"])
start_idx = calculate_start_omega(margins, y_train)
omega_idx, history = greedy_addition(X_train, y_train, start_idx)
print_prototypes(start_idx, omega_idx, history, y_train, class_names)


# 8 – 1NN с отбором эталонов и без
pred_full = calculate_predict(X_train, y_train, X_test, 1, "knn")
pred_omega = calculate_predict(X_train[omega_idx], y_train[omega_idx], X_test, 1, "knn")
compare_rows = [
    ("своя", f"1NN, вся выборка ({len(X_train)})", 1, evaluate(y_test, pred_full, n_classes)),
    ("своя", f"1NN, эталоны ({len(omega_idx)})", 1, evaluate(y_test, pred_omega, n_classes)),
]
print_metrics("1NN с отбором эталонов и без", compare_rows, len(X_test))


# 4, 7 – графики
if SHOW_PLOTS:
    plot_loo_risk(all_loo, best_k, k_values)
    errors_idx = find_errors(X_train, y_train, omega_idx)
    plot_prototype_map(X_train, y_train, omega_idx, len(start_idx), errors_idx, class_names)
    plot_ccv_history(history, len(start_idx), len(X_train))
