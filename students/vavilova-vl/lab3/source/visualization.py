from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
AXIS = "#b9b8b1"
NEGATIVE = "#2a78d6"
POSITIVE = "#eb6834"
REFERENCE = "#eda100"
MARGIN_LINE = "#7d7c77"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "grid.color": GRID,
        "legend.frameon": False,
        "font.size": 10,
        "axes.titlesize": 10.5,
    }
)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_decision_boundaries(
    panels: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    class_names: dict[int, str],
    title: str,
    axis_labels: tuple[str, str],
    path: Path,
) -> None:
    """Разделяющие поверхности на плоскости. Каждая панель: title, own, reference."""
    padding = 0.6
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    figure, axes = plt.subplots(1, len(panels), figsize=(16, 6), sharex=True, sharey=True)
    for axis, panel in zip(axes, panels):
        own, reference = panel["own"], panel["reference"]
        own_scores = own.decision_function(grid).reshape(grid_x.shape)
        reference_scores = reference.decision_function(grid).reshape(grid_x.shape)

        axis.contourf(
            grid_x, grid_y, own_scores,
            levels=[-np.inf, 0.0, np.inf], colors=[NEGATIVE, POSITIVE], alpha=0.11,
        )
        axis.contour(
            grid_x, grid_y, own_scores,
            levels=[-1.0, 1.0], colors=MARGIN_LINE, linestyles="--", linewidths=1.0,
        )
        axis.contour(grid_x, grid_y, own_scores, levels=[0.0], colors=TEXT_PRIMARY, linewidths=2.4)
        axis.contour(
            grid_x, grid_y, reference_scores,
            levels=[0.0], colors=REFERENCE, linestyles=":", linewidths=2.0,
        )

        for label, color in ((-1, NEGATIVE), (1, POSITIVE)):
            mask = y == label
            axis.scatter(
                X[mask, 0], X[mask, 1], s=16, color=color,
                edgecolors=SURFACE, linewidths=0.6, zorder=3,
            )
        axis.scatter(
            X[own.support_, 0], X[own.support_, 1], s=52,
            facecolors="none", edgecolors=TEXT_PRIMARY, linewidths=0.9, zorder=4,
        )
        axis.set_title(panel["title"])
        axis.set_xlabel(axis_labels[0])
    axes[0].set_ylabel(axis_labels[1])

    handles = [
        Line2D([], [], marker="o", linestyle="", color=NEGATIVE, label=f"{class_names[-1]} (-1)"),
        Line2D([], [], marker="o", linestyle="", color=POSITIVE, label=f"{class_names[1]} (+1)"),
        Line2D(
            [], [], marker="o", linestyle="", markersize=9, markerfacecolor="none",
            markeredgecolor=TEXT_PRIMARY, label="Опорные объекты (λ > 0)",
        ),
        Line2D([], [], color=TEXT_PRIMARY, linewidth=2.4, label="Граница f(x) = 0, своя реализация"),
        Line2D([], [], color=MARGIN_LINE, linestyle="--", linewidth=1.0, label="Границы зазора f(x) = ±1"),
        Line2D([], [], color=REFERENCE, linestyle=":", linewidth=2.0, label="Граница f(x) = 0, sklearn SVC"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0.1, 1, 0.97))
    _save(figure, path)


def plot_linear_weights(
    feature_names: list[str], own: np.ndarray, reference: np.ndarray, path: Path
) -> None:
    positions = np.arange(len(feature_names))
    height = 0.36

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.barh(positions - height / 2 - 0.02, own, height, color=NEGATIVE, label="Своя реализация")
    axis.barh(positions + height / 2 + 0.02, reference, height, color=REFERENCE, label="sklearn SVC")
    axis.axvline(0.0, color=AXIS, linewidth=1.0)
    axis.set_yticks(positions, feature_names)
    axis.invert_yaxis()
    axis.set_xlabel("Вес признака в w (признаки стандартизованы)")
    axis.set_title("Веса линейного SVM: w = Σ λᵢ yᵢ xᵢ")
    axis.grid(axis="x", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    figure.tight_layout()
    _save(figure, path)


def plot_margins(samples: list[tuple[str, np.ndarray]], title: str, path: Path) -> None:
    """Гистограммы отступов M = y * f(x), по одной панели на выборку."""
    low = min(margins.min() for _, margins in samples)
    high = max(margins.max() for _, margins in samples)
    bins = np.linspace(low - 0.1, high + 0.1, 41)

    figure, axes = plt.subplots(1, len(samples), figsize=(12, 4.6), sharex=True)
    for axis, (name, margins) in zip(axes, samples):
        axis.hist(margins, bins=bins, color=NEGATIVE, edgecolor=SURFACE, linewidth=0.8)
        axis.axvline(0.0, color=TEXT_PRIMARY, linewidth=1.4)
        axis.axvline(1.0, color=MARGIN_LINE, linewidth=1.2, linestyle="--")
        errors = int((margins < 0).sum())
        inside = int(((margins >= 0) & (margins < 1 - 1e-6)).sum())
        axis.set_title(
            f"{name}: {len(margins)} объектов\n"
            f"ошибок (M < 0): {errors}, внутри зазора (0 ≤ M < 1): {inside}"
        )
        axis.set_xlabel("Отступ M = y · f(x)")
        axis.grid(axis="y", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Количество объектов")

    handles = [
        Line2D([], [], color=TEXT_PRIMARY, linewidth=1.4, label="M = 0, граница классов"),
        Line2D([], [], color=MARGIN_LINE, linewidth=1.2, linestyle="--", label="M = 1, граница зазора"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save(figure, path)


def plot_score_comparison(panels: list[dict], y: np.ndarray, class_names: dict[int, str], path: Path) -> None:
    """Значения f(x) своей реализации против sklearn на тесте. Панель: title, own, reference."""
    figure, axes = plt.subplots(1, len(panels), figsize=(15, 6.4))
    for axis, panel in zip(axes, panels):
        own, reference = panel["own"], panel["reference"]
        low = min(own.min(), reference.min()) - 0.2
        high = max(own.max(), reference.max()) + 0.2
        axis.plot([low, high], [low, high], color=AXIS, linewidth=1.2, zorder=1)
        for label, color in ((-1, NEGATIVE), (1, POSITIVE)):
            mask = y == label
            axis.scatter(
                reference[mask], own[mask], s=22, color=color,
                edgecolors=SURFACE, linewidths=0.6, zorder=3,
            )
        axis.set_title(f"{panel['title']}\nmax |f − f_sklearn| = {np.abs(own - reference).max():.1e}")
        axis.set_xlabel("f(x), sklearn SVC")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal")
        axis.grid(linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("f(x), своя реализация")

    handles = [
        Line2D([], [], marker="o", linestyle="", color=NEGATIVE, label=f"{class_names[-1]} (-1)"),
        Line2D([], [], marker="o", linestyle="", color=POSITIVE, label=f"{class_names[1]} (+1)"),
        Line2D([], [], color=AXIS, linewidth=1.2, label="Диагональ: значения совпадают"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3)
    figure.suptitle("Значения решающей функции на тестовой выборке", fontsize=13)
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save(figure, path)
