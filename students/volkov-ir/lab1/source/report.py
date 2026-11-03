import numpy as np
import matplotlib.pyplot as plt

from model import calculate_margin
from metric import (calculate_accuracy, confusion_matrix, calculate_precision,
                    calculate_recall, calculate_f1, calculate_roc)

# Папка для сохранения графиков (для README). None — графики только показываются.
SAVE_DIR = None


def _finish(file_name):
    """Сохраняет текущий график в SAVE_DIR (если задана) и показывает его."""
    if SAVE_DIR is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(SAVE_DIR / file_name, dpi=110, bbox_inches="tight")
    plt.show()


# ---------- метрики ----------

def calculate_auc(X, y, w):
    TPR_list, FPR_list = calculate_roc(X, y, w)
    fpr = [0] + FPR_list[::-1]
    tpr = [0] + TPR_list[::-1]
    return np.trapezoid(tpr, fpr)


def evaluate(X_train, y_train, X_test, y_test, w):
    matrix, TP, TN, FP, FN = confusion_matrix(X_test, y_test, w)
    precision = calculate_precision(TP, FP) if TP + FP > 0 else 0.0
    recall = calculate_recall(TP, FN) if TP + FN > 0 else 0.0
    f1 = calculate_f1(precision, recall) if precision + recall > 0 else 0.0
    return {
        "acc_train": calculate_accuracy(X_train, y_train, w),
        "acc_test": calculate_accuracy(X_test, y_test, w),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": calculate_auc(X_test, y_test, w),
        "matrix": matrix,
    }


# ---------- вывод в консоль ----------

def print_run(title, run, metrics, feature_names):
    print("=" * 60)
    print(title)
    print("=" * 60)
    if run.get("iters") is not None:
        print(f"Итераций: {run['iters']}")
    if run.get("Q") is not None:
        print(f"Финальное Q: {run['Q']:.6f}")
    if "all_acc_train" in run:
        accs = ", ".join(f"{a:.3f}" for a in run["all_acc_train"])
        print(f"Accuracy на train по запускам мультистарта: {accs}")

    print(f"Accuracy train: {metrics['acc_train']:.4f}   test: {metrics['acc_test']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}   Recall: {metrics['recall']:.4f}   "
          f"F1: {metrics['f1']:.4f}   AUC: {metrics['auc']:.4f}")

    print("\nВеса:")
    has_start = "w_start" in run
    header = f"  {'признак':<14}{'старт':>10}{'финал':>10}" if has_start else f"  {'признак':<14}{'финал':>10}"
    print(header)
    for i, name in enumerate(feature_names):
        if has_start:
            print(f"  {name:<14}{run['w_start'][i]:>10.4f}{run['w'][i]:>10.4f}")
        else:
            print(f"  {name:<14}{run['w'][i]:>10.4f}")
    print()


def print_summary(all_metrics, all_runs):
    """Итоговая таблица: одна строка — один способ обучения."""
    print("=" * 92)
    print("ИТОГОВОЕ СРАВНЕНИЕ (метрики на test, кроме acc train)")
    print("=" * 92)
    print(f"{'метод':<28}{'acc train':>10}{'acc test':>10}{'precision':>11}"
          f"{'recall':>9}{'F1':>8}{'AUC':>8}{'итераций':>10}")
    for name, m in all_metrics.items():
        iters = all_runs[name].get("iters")
        iters = "-" if iters is None else str(iters)
        print(f"{name:<28}{m['acc_train']:>10.4f}{m['acc_test']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>8.4f}{m['auc']:>8.4f}{iters:>10}")
    print()


# ---------- графики ----------

def plot_margins(all_runs, X_train, y_train, X_test, y_test, file_name="margins.png"):
    """Отсортированные отступы: строка — метод, слева train, справа test.
    Красная зона — объект классифицирован неверно (M < 0),
    жёлтая — верно, но близко к границе (0 <= M < 1),
    зелёная — уверенно верно (M >= 1)."""
    n = len(all_runs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)

    for row, (name, run) in enumerate(all_runs.items()):
        for col, (X, y, part) in enumerate([(X_train, y_train, "train"), (X_test, y_test, "test")]):
            ax = axes[row][col]
            M = np.sort(calculate_margin(X, y, run["w"]))
            idx = np.arange(len(M))
            ax.plot(idx, M, color="black", linewidth=1)
            ax.fill_between(idx, M, 0, where=M < 0, color="tab:red", alpha=0.4, label="ошибка (M<0)")
            ax.fill_between(idx, M, 0, where=(M >= 0) & (M < 1), color="gold", alpha=0.4, label="у границы (0≤M<1)")
            ax.fill_between(idx, M, 0, where=M >= 1, color="tab:green", alpha=0.4, label="уверенно (M≥1)")
            ax.axhline(0, color="gray", linewidth=0.8)
            errors = np.sum(M < 0)
            ax.set_title(f"{name} — {part} (ошибок: {errors} из {len(M)})")
            ax.set_xlabel("объекты, отсортированные по отступу")
            ax.set_ylabel("отступ M")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    _finish(file_name)


def plot_margins_start_vs_final(all_runs, X_train, y_train, X_test, y_test, file_name="margins_start_vs_final.png"):
    """Отступы до и после обучения: серая линия - стартовые веса, чёрная - итоговые.
    Показывает, что именно делает обучение: поднимает отступы вверх и сокращает число ошибок.
    Берутся только методы, у которых известны стартовые веса (9.1, 9.2, 9.3)."""
    runs = {name: run for name, run in all_runs.items() if "w_start" in run}
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)

    for row, (name, run) in enumerate(runs.items()):
        for col, (X, y, part) in enumerate([(X_train, y_train, "train"), (X_test, y_test, "test")]):
            ax = axes[row][col]
            M_start = np.sort(calculate_margin(X, y, run["w_start"]))
            M_final = np.sort(calculate_margin(X, y, run["w"]))
            idx = np.arange(len(M_final))
            ax.plot(idx, M_start, color="gray", linestyle="--", label="старт")
            ax.plot(idx, M_final, color="black", label="после обучения")
            ax.fill_between(idx, M_final, 0, where=M_final < 0, color="tab:red", alpha=0.3)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.axhline(1, color="tab:green", linewidth=0.8, linestyle=":", label="M = 1 (минимум потерь)")
            err_start, err_final = np.sum(M_start < 0), np.sum(M_final < 0)
            ax.set_title(f"{name} - {part} (ошибок: {err_start} -> {err_final} из {len(M_final)})")
            ax.set_xlabel("объекты, отсортированные по отступу")
            ax.set_ylabel("отступ M")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    _finish(file_name)


def plot_roc(all_runs, X_test, y_test, file_name="roc.png"):
    plt.figure(figsize=(6, 6))
    for name, run in all_runs.items():
        TPR_list, FPR_list = calculate_roc(X_test, y_test, run["w"])
        auc = calculate_auc(X_test, y_test, run["w"])
        plt.plot([0] + FPR_list[::-1], [0] + TPR_list[::-1], label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайное угадывание")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые на test")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _finish(file_name)


def plot_confusion_matrices(all_metrics, file_name="confusion.png"):
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), squeeze=False)
    for ax, (name, m) in zip(axes[0], all_metrics.items()):
        matrix = m["matrix"]  # [[TP, FN], [FP, TN]] — так её строит confusion_matrix
        ax.imshow(matrix, cmap="Blues")
        labels = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{matrix[i][j]}", ha="center", va="center",
                        color="white" if matrix[i][j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["+1", "-1"])
        ax.set_yticks([0, 1], ["+1", "-1"])
        ax.set_xlabel("предсказание")
        ax.set_ylabel("на самом деле")
        ax.set_title(name, fontsize=9)
    fig.suptitle("Матрицы ошибок на test")
    fig.tight_layout()
    _finish(file_name)


def plot_weights(all_runs, feature_names, file_name="weights.png"):
    """Веса разных методов рядом — видно, какие признаки модель считает важными."""
    n = len(all_runs)
    width = 0.8 / n
    x = np.arange(len(feature_names))
    plt.figure(figsize=(12, 4))
    for k, (name, run) in enumerate(all_runs.items()):
        plt.bar(x + k * width - 0.4 + width / 2, run["w"], width=width, label=name)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.ylabel("вес")
    plt.title("Веса признаков")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _finish(file_name)


def plot_multistart(run, file_name="multistart.png"):
    """9.2: accuracy на train у каждого случайного старта — видно, насколько результат зависит от старта."""
    accs = run["all_acc_train"]
    best = int(np.argmax(accs))
    colors = ["tab:green" if i == best else "tab:blue" for i in range(len(accs))]
    plt.figure(figsize=(6, 3.5))
    plt.bar(range(1, len(accs) + 1), accs, color=colors)
    plt.ylim(min(accs) - 0.05, min(1.0, max(accs) + 0.05))
    plt.xlabel("номер старта")
    plt.ylabel("accuracy на train")
    plt.title("9.2: мультистарт (зелёный — выбранный лучший)")
    plt.tight_layout()
    _finish(file_name)
