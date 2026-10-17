import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_margins(margins, title, path):
    plt.figure(figsize=(8, 5))
    plt.hist(margins[margins > 0], bins=30, alpha=0.7, label="M > 0")
    plt.hist(margins[margins <= 0], bins=30, alpha=0.7, label="M <= 0")
    plt.axvline(0, ls="--", c="k")
    plt.xlabel("Отступ M = y(w·x + b)")
    plt.ylabel("Число объектов")
    plt.title(title)
    plt.legend()
    _save(path)


def plot_convergence(histories, ref_q, path):
    plt.figure(figsize=(9, 5))
    for name, h in histories.items():
        plt.plot(h["full_q"], label=name)
    plt.axhline(ref_q, ls="--", c="gray", label="Ridge (аналитика)")
    plt.xlabel("Эпоха / итерация")
    plt.ylabel("Q (train)")
    plt.title("Сходимость методов")
    plt.legend()
    plt.grid(alpha=0.3)
    _save(path)


def plot_recurrent(history, n_objects, path):
    full = history["full_q"]
    plt.figure(figsize=(9, 5))
    plt.plot(history["recurrent_q"], alpha=0.7, label="рекуррентная оценка Q")
    plt.plot(np.arange(1, len(full)) * n_objects, full[1:], label="полный Q (раз в эпоху)")
    plt.xlabel("Шаг SGD")
    plt.ylabel("Q")
    plt.title("Рекуррентная оценка функционала")
    plt.legend()
    plt.grid(alpha=0.3)
    _save(path)


def plot_multistart(objectives, path):
    plt.figure(figsize=(8, 5))
    for name, values in objectives.items():
        plt.plot(range(len(values)), values, marker="o", label=name)
    plt.xlabel("Seed")
    plt.ylabel("Q (validation)")
    plt.title("Мультистарт")
    plt.legend()
    plt.grid(alpha=0.3)
    _save(path)


def plot_confusion(cm, title, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xticks([0, 1], ["-1", "+1"])
    plt.yticks([0, 1], ["-1", "+1"])
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.title(title)
    _save(path)
