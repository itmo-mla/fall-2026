import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model import calculate_knn
from metric import calculate_accuracy, calculate_macro_f1

# папка для сохранения графиков (для README), None – графики только показываются
SAVE_DIR = None
# до какого k рисуется второй, увеличенный график LOO
K_ZOOM = 50

METHOD_NAMES = {"knn": "KNN", "parzen": "окно Парзена, сумма по k соседям", "parzen_lecture": "окно Парзена, сумма по всей выборке (лекция)", "knn_distance": "KNN, вес 1/расстояние"}
# короткие названия для таблиц
SHORT_NAMES = {"knn": "KNN", "parzen": "окно Парзена (k ближайших)", "parzen_lecture": "окно Парзена (лекция)", "knn_distance": "KNN, вес 1/расстояние"}
METHOD_COLORS = {"knn": "#2a78d6", "parzen": "#eb6834", "parzen_lecture": "#1baf7a"}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def _finish(fig, file_name):
    # сохраняет график в SAVE_DIR (если задана) и показывает его
    fig.tight_layout()
    if SAVE_DIR is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_DIR / file_name, dpi=150)
    plt.show()
    plt.close(fig)


# ---------- метрики ----------

def evaluate(y_true, pred_labels, n_classes):
    return {
        "accuracy": calculate_accuracy(y_true, pred_labels),
        "f1": calculate_macro_f1(y_true, pred_labels, n_classes),
        "errors": int(np.sum(pred_labels != y_true)),
    }


def find_errors(X_train, y_train, omega_idx):
    # объекты train, на которых 1NN по эталонам ошибается (без самого объекта, как в CCV)
    errors_idx = []
    for i in range(0, len(X_train)):
        rest_idx = omega_idx[omega_idx != i]
        if calculate_knn(X_train[rest_idx], y_train[rest_idx], X_train[i], 1) != y_train[i]:
            errors_idx.append(i)
    return np.array(errors_idx, dtype=int)


# ---------- вывод в консоль ----------

def print_title(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_loo(all_loo, best_k, k_values, n_train):
    print_title("Подбор k по LOO на train")
    for method, loo_errors in all_loo.items():
        loo_errors = np.array(loo_errors)
        minima = [k for k, e in zip(k_values, loo_errors) if e == loo_errors.min()]
        print(f"{METHOD_NAMES[method]:<46} лучшее k = {best_k[method]:<3} LOO = {loo_errors.min():.4f} (ошибок {round(loo_errors.min() * n_train)} из {n_train}), минимум при k = {minima}")
    print()


# строка таблицы – (реализация, модель, k, метрики)
def print_metrics(title, rows, n_test):
    print_title(title)
    print(f"{'реализация':<12}{'модель':<30}{'k':>4}{'accuracy':>10}{'macro-F1':>10}{'ошибок':>8}")
    for impl, name, k, metrics in rows:
        print(f"{impl:<12}{name:<30}{k:>4}{metrics['accuracy']:>10.3f}{metrics['f1']:>10.3f}{metrics['errors']:>8}")
    print(f"(ошибок из {n_test})\n")


def print_sklearn_check(all_loo, sklearn_results):
    print_title("Проверка реализации: LOO своих функций и sklearn")
    for method in ["knn", "parzen", "parzen_lecture"]:
        sklearn_loo = sklearn_results[method]["loo"]
        own_loo = np.array(all_loo[method][:len(sklearn_loo)])
        n_same = int(np.sum(np.isclose(own_loo, sklearn_loo)))
        print(f"{METHOD_NAMES[method]:<46} совпадает при {n_same} из {len(sklearn_loo)} k")
    print()


def print_prototypes(start_idx, omega_idx, history, y_train, class_names):
    print_title("Отбор эталонов")
    n_errors = [round(ccv * len(y_train)) for ccv in history]
    counts = np.bincount(y_train[omega_idx], minlength=len(class_names))
    print(f"стартовые эталоны: {start_idx.tolist()}")
    print(f"итоговые эталоны:  {omega_idx.tolist()}, всего {len(omega_idx)} из {len(y_train)}")
    print(f"по видам: {dict(zip(class_names, counts.tolist()))}")
    print(f"ошибок CCV по шагам (из {len(y_train)}): {n_errors}\n")


# ---------- графики ----------

def plot_loo_risk(all_loo, best_k, k_values):
    # два графика, для всех k и увеличенный для k до K_ZOOM, где находится минимум
    _plot_loo_risk(all_loo, best_k, k_values, k_values[-1], "все k", "loo_risk_all.png")
    _plot_loo_risk(all_loo, best_k, k_values, K_ZOOM, f"k от 1 до {K_ZOOM}", "loo_risk_zoom.png")


def _plot_loo_risk(all_loo, best_k, k_values, k_max, title, file_name):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ks = [k for k in k_values if k <= k_max]
    for method, loo_errors in all_loo.items():
        loss = np.array(loo_errors[:len(ks)]) * 100
        ax.plot(ks, loss, color=METHOD_COLORS[method], linewidth=2, label=f"{METHOD_NAMES[method]} (k* = {best_k[method]})")
        ax.scatter([best_k[method]], [loss[best_k[method] - 1]], s=64, color=METHOD_COLORS[method], edgecolors="white", linewidths=2, zorder=3)
    ax.set_title(f"Эмпирический риск LOO, {title}")
    ax.set_xlabel("число соседей k")
    ax.set_ylabel("ошибка LOO, %")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    _finish(fig, file_name)


def plot_prototype_map(X_train, y_train, omega_idx, n_start, errors_idx, class_names, file_name="prototypes_map.png"):
    # PCA только для картинки, 4 признака сводятся к 2, классификация идет в исходных 4
    pca = PCA(n_components=2).fit(X_train)
    Z = pca.transform(X_train)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for c in range(0, len(class_names)):
        ax.scatter(*Z[y_train == c].T, s=18, color=CLASS_COLORS[c], alpha=0.35, label=class_names[c])
    start_idx, added_idx = omega_idx[:n_start], omega_idx[n_start:]
    ax.scatter(*Z[start_idx].T, s=170, marker="s", c=[CLASS_COLORS[c] for c in y_train[start_idx]], edgecolors="black", linewidths=1.5, zorder=3)
    ax.scatter(*Z[added_idx].T, s=170, marker="o", c=[CLASS_COLORS[c] for c in y_train[added_idx]], edgecolors="black", linewidths=1.5, zorder=3)
    # пустые маркеры только для легенды, цвет эталона уже показывает его вид
    ax.scatter([], [], s=170, marker="s", facecolor="white", edgecolors="black", linewidths=1.5, label="эталон, стартовый")
    ax.scatter([], [], s=170, marker="o", facecolor="white", edgecolors="black", linewidths=1.5, label="эталон, добавлен (номер шага)")
    # номер шага, на котором эталон добавлен
    for step, i in enumerate(added_idx, start=1):
        ax.annotate(str(step), Z[i], xytext=(7, 7), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.scatter(*Z[errors_idx].T, s=90, marker="x", color="black", linewidths=2, zorder=4, label="ошибка 1NN по эталонам")
    ax.set_title(f"Эталоны: {len(omega_idx)} из {len(X_train)} объектов train")
    ax.set_xlabel(f"главная компонента 1 ({var[0]:.0f}% дисперсии)")
    ax.set_ylabel(f"главная компонента 2 ({var[1]:.0f}% дисперсии)")
    ax.grid(alpha=0.3)
    # легенда справа от графика, чтобы не закрывать точки
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _finish(fig, file_name)


def plot_ccv_history(history, n_start, n_train, file_name="prototypes_ccv.png"):
    n_errors = np.round(np.array(history) * n_train).astype(int)
    n_protos = np.arange(n_start, n_start + len(history))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_protos, n_errors, color="#2a78d6", linewidth=2, marker="o", markersize=8)
    for x, e in zip(n_protos, n_errors):
        ax.annotate(str(e), (x, e), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title("Жадное добавление: ошибки 1NN по эталонам (CCV)")
    ax.set_xlabel("число эталонов")
    ax.set_ylabel(f"ошибок на train из {n_train}")
    ax.set_xticks(n_protos)
    ax.set_ylim(0, n_errors.max() * 1.15)
    ax.grid(alpha=0.3)
    _finish(fig, file_name)
