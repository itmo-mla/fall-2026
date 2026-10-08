from pathlib import Path

import numpy as np
import report
from sklearn_baseline import run_sklearn_baseline
from task import run_correlation_init, run_multistart, run_sampling_comparison
from initialization import calculate_w_random, calculate_q_start
from model import calculate_sgd
from metric import calculate_accuracy
from data_preprocessing import train_and_test, standardize, X, y, feature_names
from report import (evaluate, print_run, print_summary, plot_margins, plot_margins_start_vs_final, plot_multistart,
                    plot_roc, plot_confusion_matrices, plot_weights)
np.random.seed(42)  # фиксируем случайность, чтобы результаты совпадали с README

report.SAVE_DIR = Path(__file__).parent.parent / "images"  # графики для README

test_size = 0.3
X_train, y_train, X_test, y_test = train_and_test(X, y, test_size)

scale_columns = [feature_names.index('Age'), feature_names.index('Fare')]
X_train, X_test = standardize(X_train, X_test, scale_columns)


# momentum - вклад предыдущего направления обновления весов. Чем больше значение, тем сильнее инерция.
# q_update_rate - определяет, насколько сильно текущая ошибка объекта влияет на рекурсивную оценку качества Q
# l2_strength - коэффициент L2-регуляризации
# margin_epsilon - технический стабилизатор для margin-based sampling
# tolerance - насколько мало должно измениться Q за одну эпоху (len(X) шагов), чтобы эпоха считалась "спокойной"; 3 спокойные эпохи подряд - остановка
# q_start_size - количество случайных объектов, по которым рассчитывается стартовое значение Q
# n_stars - число случайных инициализаций в multistart
# learning_rate - доля от шага скорейшего спуска h: 1 - полностью подогнать веса под текущий объект, 0.01 - сделать 1% этой поправки (плавное обучение)
momentum = 0.5
q_update_rate = 0.001
l2_strength = 1e-3
margin_epsilon = 1e-5
tolerance = 1e-2
q_start_size = 50
n_starts = 5
learning_rate = 1e-2

SHOW_PLOTS = True

all_runs = {}  # название метода -> результат обучения (веса, Q, итерации)

# 9.1 — инициализация через корреляцию
print("Обучение 9.1 ...")
all_runs["9.1 корреляция"] = run_correlation_init(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, learning_rate)

# 9.2 — случайная инициализация + мультистарт
print("Обучение 9.2 ...")
all_runs["9.2 мультистарт"] = run_multistart(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, n_starts, learning_rate)

# 9.3 — один и тот же случайный старт, два способа предъявления объектов
print("Обучение 9.3 ...")
w_start_fixed = calculate_w_random(X_train.shape[1])
run_margin, run_uniform = run_sampling_comparison(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, w_start_fixed, learning_rate)
all_runs["9.3 по модулю отступа"] = run_margin
all_runs["9.3 случайно"] = run_uniform

#11 — эталон
print("Обучение sklearn ...\n")
all_runs["sklearn SGDClassifier"] = run_sklearn_baseline(X_train, y_train, l2_strength)


# 10 — оценка качества
all_metrics = {}
for name, run in all_runs.items():
    all_metrics[name] = evaluate(X_train, y_train, X_test, y_test, run["w"])
    print_run(name, run, all_metrics[name], feature_names)

print_summary(all_metrics, all_runs)

if SHOW_PLOTS:
    plot_margins(all_runs, X_train, y_train, X_test, y_test)  # 2 — отступы
    plot_margins_start_vs_final(all_runs, X_train, y_train, X_test, y_test)  # отступы до и после обучения
    if "9.2 мультистарт" in all_runs:  # график есть только если 9.2 запускался
        plot_multistart(all_runs["9.2 мультистарт"])
    plot_roc(all_runs, X_test, y_test)
    plot_confusion_matrices(all_metrics)
    plot_weights(all_runs, feature_names)


# Дополнительно: влияние инерции. Один и тот же старт, по 3 запуска на каждое значение momentum
print("Влияние инерции:")
w0 = calculate_w_random(X_train.shape[1])
for gamma in [0.0, 0.5, 0.9]:
    acc_tr, acc_te, iters = [], [], []
    for seed in range(3):
        np.random.seed(seed)
        Q0 = calculate_q_start(X_train, y_train, w0, q_start_size)
        w, Q, c = calculate_sgd(X_train, y_train, w0.copy(), gamma, q_update_rate, Q0, l2_strength, margin_epsilon, tolerance, learning_rate, sampling="random")
        acc_tr.append(calculate_accuracy(X_train, y_train, w))
        acc_te.append(calculate_accuracy(X_test, y_test, w))
        iters.append(c)
    print(f"momentum={gamma}: acc train {np.mean(acc_tr):.3f}, acc test {np.mean(acc_te):.3f}, итераций в среднем {np.mean(iters):.0f}")
