from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from svm import SVM

COLORS = {-1: "#D95459", 1: "#3268BF"}
CLASS_LABELS = {-1: "−1", 1: "+1"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_grid(X: np.ndarray, step_count: int = 175
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сетка точек вокруг облака данных для отрисовки границы решения"""
    dx, dy = np.ptp(X, axis=0)
    padx, pady = max(0.65, dx * 0.09), max(0.65, dy * 0.09)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - padx, X[:, 0].max() + padx, step_count),
        np.linspace(X[:, 1].min() - pady, X[:, 1].max() + pady, step_count),
    )
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def plot_data_projection(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """PCA-проекция обучающих объектов на две главные компоненты"""
    centered = X - X.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:2].T
    share = singular[:2] ** 2 / np.sum(singular ** 2)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for cls in (-1, 1):
        mask = y == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=28, c=COLORS[cls],
            edgecolors="white", linewidths=0.5, alpha=0.82,
            label=CLASS_LABELS[cls], zorder=4,
        )
    ax.set_xlabel(f"Главная компонента 1 ({100 * share[0]:.1f}%)")
    ax.set_ylabel(f"Главная компонента 2 ({100 * share[1]:.1f}%)")
    ax.set_title("PCA-проекция обучающей выборки (Pima Indians Diabetes)")
    ax.legend(title="Класс", loc="best", framealpha=0.96)
    ax.grid(True, linestyle="--", alpha=0.25)
    _save(fig, path)


def plot_boundary(ax: plt.Axes, model: SVM, X: np.ndarray, y: np.ndarray,
                  title: str, grid: tuple | None = None,
                  show_support: bool = True, legend: bool = True) -> None:
    """Заливка областей, граница f(x)=0, отступы ±1 и опорные векторы"""
    if grid is None:
        grid = make_grid(X)
    xx, yy, points = grid
    values = model.decision_function(points).reshape(xx.shape)

    ax.contourf(xx, yy, values >= 0, levels=[-0.5, 0.5, 1.5],
                colors=["#f7dfe0", "#dce7fa"], alpha=0.78, zorder=0)
    if values.min() < 0 < values.max():
        ax.contour(xx, yy, values, levels=[0], colors=["#202020"],
                   linewidths=1.6, zorder=2)
    margins = [level for level in (-1, 1) if values.min() < level < values.max()]
    if margins:
        ax.contour(xx, yy, values, levels=margins, colors=["#606874"],
                   linestyles="--", linewidths=1.1, zorder=1)

    for cls in (-1, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=27, c=COLORS[cls],
                   edgecolors="white", linewidths=0.5, alpha=0.82, zorder=4)
    if show_support:
        support = model.support_vectors
        ax.scatter(support[:, 0], support[:, 1], s=77, facecolors="none",
                   edgecolors="black", linewidths=1.1, zorder=6)

    ax.set_title(title, fontsize=11.5, fontweight="semibold", pad=10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(True, linestyle="--", alpha=0.15)
    if legend:
        ax.legend(handles=_legend_handles(show_support), loc="upper right",
                  fontsize=8.2, framealpha=0.96, facecolor="white")


def _legend_handles(show_support: bool) -> list:
    handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=COLORS[-1], markeredgecolor="white",
               markersize=7, label="Класс −1"),
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=COLORS[1], markeredgecolor="white",
               markersize=7, label="Класс +1"),
        Patch(facecolor="#f7dfe0", edgecolor="none", label="Область −1"),
        Patch(facecolor="#dce7fa", edgecolor="none", label="Область +1"),
        Line2D([], [], c="#202020", linewidth=1.6, label="Граница f(x)=0"),
        Line2D([], [], c="#606874", linestyle="--", linewidth=1.1,
               label="Отступ f(x)=±1"),
    ]
    if show_support:
        handles.append(Line2D([], [], linestyle="none", marker="o",
                              markersize=8, markerfacecolor="none",
                              markeredgecolor="black", label="Опорный вектор"))
    return handles


def plot_single_boundary(model: SVM, X: np.ndarray, y: np.ndarray,
                         title: str, path: Path) -> None:
    """Одна панель с границей решения выбранной модели"""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    plot_boundary(ax, model, X, y, title)
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    _save(fig, path)


def plot_kernel_comparison(models: list[SVM], X: np.ndarray, y: np.ndarray,
                           titles: list[str], path: Path) -> None:
    """Две границы рядом: линейное и RBF-ядро в одних осях"""
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0), sharex=True,
                             sharey=True, constrained_layout=True)
    grid = make_grid(X)
    for ax, model, title in zip(axes, models, titles):
        plot_boundary(ax, model, X, y, title, grid=grid,
                      legend=(ax is axes[0]))
    for ax in axes:
        ax.set_xlabel("Главная компонента 1")
    axes[0].set_ylabel("Главная компонента 2")
    fig.suptitle("Сравнение ядер на PCA-проекции", fontsize=14,
                 fontweight="bold")
    _save(fig, path)


def plot_c_effect(results: list[dict], path: Path) -> None:
    """Влияние C на точность и число опорных векторов"""
    c = np.array([row["C"] for row in results])
    train_acc = np.array([row["train_acc"] for row in results])
    test_acc = np.array([row["test_acc"] for row in results])
    support = np.array([row["support"] for row in results])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14.5, 5.5), constrained_layout=True
    )
    left.semilogx(c, train_acc, "o-", linewidth=2, c="#3268BF",
                  label="Train accuracy")
    left.semilogx(c, test_acc, "s-", linewidth=2, c="#D95459",
                  label="Test accuracy")
    for x, value in zip(c, test_acc):
        left.annotate(f"{value:.3f}", (x, value), xytext=(0, -14),
                      textcoords="offset points", fontsize=7.5, ha="center")
    left.set_xlabel("C (логарифмическая шкала)")
    left.set_ylabel("Accuracy")
    left.set_title("Точность RBF SVM")
    left.legend(loc="best", framealpha=0.96)
    left.grid(True, alpha=0.25)

    right.semilogx(c, support, "D-", linewidth=2, c="#238267")
    for x, value in zip(c, support):
        right.annotate(str(int(value)), (x, value), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=8.5)
    right.set_xlabel("C (логарифмическая шкала)")
    right.set_ylabel("Число опорных векторов")
    right.set_title("Сложность модели")
    right.grid(True, alpha=0.25)

    fig.suptitle("Влияние параметра C на RBF-ядро", fontsize=13,
                 fontweight="bold")
    _save(fig, path)


def plot_confusion_matrices(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                            path: Path) -> None:
    """Матрицы ошибок для четырёх моделей в одной строке"""
    from sklearn.metrics import confusion_matrix

    names = list(predictions)
    matrices = [
        confusion_matrix(y_true, predictions[name], labels=[-1, 1])
        for name in names
    ]
    vmax = max(1, max(int(matrix.max()) for matrix in matrices))
    codes = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 5.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name, matrix in zip(axes, names, matrices):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                value = int(matrix[i, j])
                correct = i == j
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, linewidth=2.5,
                    edgecolor="#198567" if correct else "#d35454",
                ))
                ax.text(j, i, f"{codes[i, j]}\n{value}", ha="center",
                        va="center", fontsize=13, fontweight="bold",
                        color="white" if value > 0.48 * vmax else "#25313e")
        ax.set_xticks([0, 1], ["−1", "+1"])
        ax.set_yticks([0, 1], ["−1", "+1"])
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title(name, fontsize=10.5, fontweight="semibold")
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Матрицы ошибок на тестовой выборке", fontsize=13,
                 fontweight="bold")
    _save(fig, path)
