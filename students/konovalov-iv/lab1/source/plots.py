import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_margins(margins_by_title, path, delta=1.0):
    fig, axes = plt.subplots(1, len(margins_by_title), figsize=(6 * len(margins_by_title), 4))
    axes = np.atleast_1d(axes)
    for ax, (title, M) in zip(axes, margins_by_title.items()):
        M = np.sort(M)
        idx = np.arange(len(M))
        zones = [
            (M <= -delta, "red", f"шумовые, M ≤ -{delta:g}"),
            (np.abs(M) < delta, "gold", f"пограничные, |M| < {delta:g}"),
            (M >= delta, "green", f"надёжные, M ≥ {delta:g}"),
        ]
        for mask, color, label in zones:
            ax.fill_between(idx, 0, M, where=mask, color=color, alpha=0.35,
                            step="mid", label=f"{label}: {mask.sum()}")
        ax.plot(idx, M, color="black", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=1)
        for level in (-delta, delta):
            ax.axhline(level, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("объекты, упорядоченные по возрастанию отступа")
        ax.set_ylabel("отступ M = ⟨w, x⟩·y")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training(models_by_label, path, title, per_epoch=False):
    fig, (ax_q, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    for label, model in models_by_label.items():
        if per_epoch:
            q = model.history["Q_epoch"]
            ax_q.plot(np.arange(1, len(q) + 1), q, label=label)
        else:
            ax_q.plot(model.history["Q"], label=label)
        epochs = np.arange(1, len(model.history["val_acc"]) + 1)
        ax_acc.plot(epochs, model.history["val_acc"], label=label)
    ax_q.set_yscale("log")
    ax_q.set_xlabel("эпоха" if per_epoch else "итерация (предъявленный объект)")
    ax_q.set_ylabel("Q в конце эпохи (лог. шкала)" if per_epoch else "рекуррентная оценка Q (лог. шкала)")
    ax_q.set_title("Сходимость функционала Q")
    ax_acc.set_xlabel("эпоха")
    ax_acc.set_ylabel("accuracy на валидации")
    ax_acc.set_title("Качество на валидационной выборке")
    ax_q.grid(alpha=0.3)
    ax_acc.grid(alpha=0.3)
    fig.suptitle(title)
    fig.legend(*ax_q.get_legend_handles_labels(), loc="lower center", ncol=len(models_by_label))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_multistart(risks, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    starts = np.arange(1, len(risks) + 1)
    best = int(np.argmin(risks))
    ax.scatter(starts, risks, s=50, color="tab:blue", label="запуск", zorder=3)
    ax.scatter(starts[best], risks[best], s=90, color="tab:orange", label="лучший запуск", zorder=4)
    ax.set_xticks(starts)
    ax.set_xlabel("номер запуска")
    ax.set_ylabel("эмпирический риск Q на обучении")
    ax.set_title("Мультистарт: случайная инициализация весов")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(results, path):
    names = list(results)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    for offset, color, metric in [(-0.12, "tab:blue", "accuracy"), (0.12, "tab:orange", "f1")]:
        values = [results[n][metric] for n in names]
        ax.scatter(values, y + offset, s=60, color=color, label=metric, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("значение метрики на тестовой выборке")
    ax.set_title("Сравнение моделей")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
