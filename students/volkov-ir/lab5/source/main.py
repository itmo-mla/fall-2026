import numpy as np
from data_preprocessing import prepare_data
import model
import metric
import sklearn_baseline
import plots

# EPSILON - порог остановки по изменению Q
# SOLVERS - sklearn: lbfgs - по умолчанию, newton-cholesky - метод Ньютона
EPSILON = 1e-9
SOLVERS = ["lbfgs", "newton-cholesky"]

SAVE_PLOTS = False


# 1 - данные
X_train, X_test, y_train, y_test, feature_names = prepare_data()
print("1. Данные")
print("train", X_train.shape, "test", X_test.shape)
print("доля класса +1: train", round(np.mean(y_train == 1), 3), "test", round(np.mean(y_test == 1), 3))
print()

# 2 - Ньютон-Рафсон из нулей и IRLS из МНК
w_newton, Q_newton, w_list_newton, Q_list_newton, count_newton = model.method_newton_raph(X_train, y_train, model.init_w_zero(X_train), EPSILON)
w_irls, Q_irls, w_list_irls, Q_list_irls, count_irls = model.method_irls(X_train, y_train, model.init_w(X_train, y_train), EPSILON)

print("2. Ньютон-Рафсон и IRLS")
print("Ньютон-Рафсон, старт из нулей: итераций", count_newton, "Q =", round(Q_newton, 6))
for t in range(0, len(Q_list_newton)):
    print("  итерация", t, "Q =", round(Q_list_newton[t], 6))
print()
print("IRLS, старт из МНК: итераций", count_irls, "Q =", round(Q_irls, 6))
for t in range(0, len(Q_list_irls)):
    print("  итерация", t, "Q =", round(Q_list_irls[t], 6))
print()

if SAVE_PLOTS:
    plots.plot_q_history(Q_list_newton, Q_list_irls, "../images/q_newton_irls.png")

# 3 - сравнение со sklearn
print("3. Сравнение со sklearn")
for solver in SOLVERS:
    w_sk, proba_sk, pred_sk, count_sk = sklearn_baseline.calculate_sklearn_logreg(X_train, y_train, X_test, solver)
    diff_w, diff_proba, count_pred = sklearn_baseline.calculate_comparison(X_test, w_newton, w_sk, proba_sk, pred_sk)
    print(solver + ", итераций:", count_sk)
    print("  Ньютон-Рафсон: веса", format(diff_w, ".1e"), "| вероятности", format(diff_proba, ".1e"), "| несовпадений", count_pred)
    diff_w, diff_proba, count_pred = sklearn_baseline.calculate_comparison(X_test, w_irls, w_sk, proba_sk, pred_sk)
    print("  IRLS:          веса", format(diff_w, ".1e"), "| вероятности", format(diff_proba, ".1e"), "| несовпадений", count_pred)
print()

# эталон для метрик - lbfgs
w_sklearn = sklearn_baseline.calculate_sklearn_logreg(X_train, y_train, X_test, "lbfgs")[0]

# 4 - метрики на test
print("4. Метрики на test")
names = ["Ньютон-Рафсон", "IRLS", "sklearn"]
weights = [w_newton, w_irls, w_sklearn]
for i in range(0, len(names)):
    accuracy = metric.calculate_accuracy(X_test, y_test, weights[i])
    matrix, TP, TN, FP, FN = metric.confusion_matrix(X_test, y_test, weights[i])
    precision = metric.calculate_precision(TP, FP)
    recall = metric.calculate_recall(TP, FN)
    f1 = metric.calculate_f1(precision, recall)
    log_loss = metric.calculate_log_loss(X_test, y_test, weights[i])
    print(names[i])
    print("  accuracy ", round(accuracy, 4))
    print("  precision", round(precision, 4))
    print("  recall   ", round(recall, 4))
    print("  f1       ", round(f1, 4))
    print("  log-loss ", round(log_loss, 4))
    print("  TP", TP, "FN", FN, "FP", FP, "TN", TN)
    print()
