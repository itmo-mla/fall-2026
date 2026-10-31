from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from data import PreparedData
from evaluation import EvaluationResults
from model import predict_labels


def experiment_label(name: str) -> str:
    variant, suffix = name.rsplit("_", 1)
    if not suffix.isdigit():
        variant = name
    labels = {
        "correlation_uniform": "Корреляция, равномерно",
        "correlation_margin": "Корреляция, по отступу",
        "random_uniform": "Случайный старт, равномерно",
        "random_margin": "Случайный старт, по отступу",
        "steepest_uniform": "Скорейший спуск",
    }
    return labels.get(variant, name)


def plot_class_distribution(distribution: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(6, 3))
    for wine_type, label, color in [
        ("white", "Белое", "#387F8C"),
        ("red", "Красное", "#B84452"),
    ]:
        axis.bar(
            label,
            distribution.loc[wine_type, "count"],
            color=color,
            label=label,
        )
    axis.legend()
    axis.set(
        xlabel="Тип вина",
        ylabel="Количество образцов",
        title="Тип вина после очистки",
    )
    return figure


def plot_initial_margins(margins_by_start: dict[str, np.ndarray]) -> Figure:
    figure, axes = plt.subplots(
        1, len(margins_by_start), figsize=(10, 3.5), squeeze=False
    )
    for axis, (name, margins) in zip(axes[0], margins_by_start.items()):
        axis.hist(margins, bins=35, color="#387F8C", label="Train")
        axis.axvline(0, color="black", linestyle=":", label="M = 0")
        axis.set(
            xlabel="Отступ M",
            ylabel="Количество объектов",
            title=f"До обучения: {name.lower()}",
        )
        axis.legend()
    return figure


def plot_final_margins(margins_by_split: dict[str, np.ndarray]) -> Figure:
    test_margins = margins_by_split["test"]
    bins = np.histogram_bin_edges(test_margins, bins=45)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    distribution_axis, test_axis = axes

    distribution_axis.hist(test_margins, bins=bins, color="#387F8C", label="Test")
    distribution_axis.hist(
        test_margins[test_margins < 0],
        bins=bins,
        color="#B84452",
        label="M < 0",
    )
    ordered_margins = np.sort(test_margins)
    object_positions = np.arange(len(ordered_margins))
    for mask, label, color in [
        (ordered_margins >= 0, "M ≥ 0", "#387F8C"),
        (ordered_margins < 0, "M < 0", "#B84452"),
    ]:
        test_axis.scatter(
            object_positions[mask],
            ordered_margins[mask],
            color=color,
            label=label,
            s=8,
        )
    distribution_axis.axvline(0, color="black", linestyle=":", label="M = 0")
    distribution_axis.axvline(1, color="gray", linestyle="--", label="Нулевая потеря: M = 1")
    test_axis.axhline(0, color="black", linestyle=":")
    distribution_axis.set(
        xlabel="Отступ M",
        ylabel="Количество объектов",
        title="Отступы после обучения: test",
    )
    test_axis.set(
        xlabel="Объекты test, отсортированные по M",
        ylabel="Отступ M",
        title=f"Отрицательных отступов: {(test_margins < 0).sum()} из {len(test_margins)}",
    )
    distribution_axis.legend()
    test_axis.legend()
    return figure


def plot_recurrent_quality(
    histories: dict[str, pd.DataFrame],
    variants: pd.DataFrame,
) -> Figure:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    variants = variants.set_index("run")
    for name, history in histories.items():
        variant = variants.loc[name]
        if name == "steepest_uniform":
            axis = axes[0, 2]
            method_label = "Скорейший спуск без инерции"
        else:
            row = 0 if variant["initialization"] == "correlation" else 1
            column = 0 if variant["sampling"] == "uniform" else 1
            axis = axes[row, column]
            method_label = "SGD с инерцией"
        if variant["initialization"] == "correlation":
            initialization_label = "Старт через корреляцию"
        else:
            initialization_label = f"Случайный старт, seed {int(variant['seed'])}"
        sampling_label = (
            "Равномерно" if variant["sampling"] == "uniform" else "По модулю отступа"
        )
        axis.plot(
            history["epoch"],
            history["train_loss"],
            label="Средняя потеря по всей train",
        )
        axis.plot(
            history["epoch"],
            history["Q"],
            label="Рекуррентная оценка Q",
        )
        axis.set(
            title=f"{method_label}\n{initialization_label} · {sampling_label}",
            xlabel="Эпоха",
            ylabel="Квадратичная потеря",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[1, 2].set_axis_off()
    figure.suptitle("Рекуррентная оценка Q и средняя потеря по всей train", fontsize=14)
    return figure


def plot_learning_curves(
    histories: dict[str, pd.DataFrame],
) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    methods = [("SGD с инерцией", False), ("Скорейший спуск", True)]
    splits = [("Train", "train_loss"), ("Validation", "validation_loss")]
    for row, (method, steepest) in enumerate(methods):
        for column, (split, metric) in enumerate(splits):
            axis = axes[row, column]
            for name, history in histories.items():
                if name.startswith("steepest") == steepest:
                    axis.plot(
                        history["epoch"], history[metric], label=experiment_label(name)
                    )
            axis.set(
                title=f"{method}: {split}",
                xlabel="Эпоха",
                ylabel="Средняя квадратичная потеря",
            )
            axis.legend(fontsize=8)
    return figure


def draw_confusion_matrix(
    axis: Axes,
    labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> None:
    predictions = predict_labels(scores)
    matrix = confusion_matrix(labels, predictions, labels=[-1, 1])
    colors = ["#D6ECEF", "#F3D1D5"]
    legend = [
        Patch(facecolor=colors[0], label="Правильный ответ"),
        Patch(facecolor=colors[1], label="Ошибка"),
    ]

    axis.imshow([[0, 1], [1, 0]], cmap=ListedColormap(colors), vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )
    axis.set(
        title=model_name,
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Белое", "Красное"],
        yticklabels=["Белое", "Красное"],
    )
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
    )


def plot_confusion_matrices(
    labels: np.ndarray,
    model_scores: dict[str, np.ndarray],
) -> Figure:
    figure, axes = plt.subplots(
        1, len(model_scores), figsize=(10, 4.5), squeeze=False
    )
    for axis, (name, scores) in zip(axes[0], model_scores.items()):
        draw_confusion_matrix(axis, labels, scores, name)
    return figure


def plot_roc_precision_recall(
    labels: np.ndarray,
    model_scores: dict[str, np.ndarray],
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    roc_axis, precision_recall_axis = axes
    red_labels = labels == 1
    for name, scores in model_scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            red_labels, scores
        )
        precision, recall, _ = precision_recall_curve(red_labels, scores)
        style = "--" if name == "RidgeClassifier" else "-"
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=name,
            linestyle=style,
        )
        precision_recall_axis.plot(recall, precision, label=name, linestyle=style)
    roc_axis.plot(
        [0, 1],
        [0, 1],
        ":",
        color="gray",
        label="Случайное ранжирование",
    )
    precision_recall_axis.axhline(
        red_labels.mean(),
        linestyle=":",
        color="gray",
        label="Доля красного вина",
    )
    roc_axis.set(
        title="ROC: test",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    precision_recall_axis.set(
        title="Precision–recall: красное вино",
        xlabel="Recall",
        ylabel="Precision",
    )
    for axis in axes:
        axis.legend(fontsize=8)
    return figure


def plot_multistart(validation_runs: pd.DataFrame, best_run: str) -> Figure:
    runs = validation_runs[validation_runs["run"] != "steepest_uniform"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    metrics = ["f1_red", "validation_loss"]
    positions = []
    labels = []
    offset = 1

    for sampling, label, color in [
        ("uniform", "Равномерное предъявление", "#387F8C"),
        ("margin", "Предъявление по модулю отступа", "#B84452"),
    ]:
        group = runs[runs["sampling"] == sampling].sort_values(
            ["initialization", "seed"]
        )
        group_positions = np.arange(len(group)) + offset
        positions.extend(group_positions)
        labels.extend(
            "Через корреляцию"
            if row.initialization == "correlation"
            else f"Случайная (seed {row.seed})"
            for row in group.itertuples()
        )
        selected = group["run"].to_numpy() == best_run

        for axis, metric in zip(axes, metrics):
            axis.text(
                0.02,
                offset - 0.7,
                label,
                transform=axis.get_yaxis_transform(),
                color=color,
                fontweight="bold",
                fontsize=10,
            )
            axis.scatter(
                group[metric],
                group_positions,
                color=color,
                label=label,
                s=40,
            )
            for position, value in zip(group_positions, group[metric]):
                axis.annotate(
                    f"{value:.4f}",
                    (value, position),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                )
            if selected.any():
                axis.scatter(
                    group.loc[selected, metric],
                    group_positions[selected],
                    marker="*",
                    color=color,
                    edgecolors="black",
                    linewidths=0.7,
                    s=160,
                    label="Выбран для сравнения с эталоном",
                )
        offset += len(group) + 2

    for axis, title, xlabel in zip(
        axes,
        ["F1 — больше лучше", "Квадратичная потеря — меньше лучше"],
        ["F1 красного класса на validation", "Средняя потеря на validation"],
    ):
        axis.set(
            title=title,
            xlabel=xlabel,
            ylabel="Инициализация весов",
            yticks=positions,
            yticklabels=labels,
            ylim=(offset - 2, -0.5),
        )
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    axes[0].set_xlim(0.95, 1.0)
    axes[1].margins(x=0.3)
    figure.suptitle("SGD с инерцией: сравнение инициализации", fontsize=13)
    return figure


def plot_weight_stability(histories: dict[str, pd.DataFrame]) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 4))
    for name, history in histories.items():
        axis.semilogy(
            history["epoch"], history["relative_weight_change"], label=experiment_label(name)
        )
    axis.axhline(1e-3, color="black", linestyle="--", label="Допуск остановки: 0.001")
    axis.set(
        title="Изменение весов между эпохами",
        xlabel="Эпоха",
        ylabel=r"$\|w_{k+1}-w_k\|/\max(1,\|w_k\|)$",
    )
    axis.legend(fontsize=7)
    return figure


def plot_sampling_distributions(margins: dict[str, np.ndarray]) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 4))
    bins = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, np.inf])
    labels = [
        "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1",
        "1–1.25", "1.25–1.5", "1.5–2", "2 и выше",
    ]
    positions = np.arange(len(labels))
    for strategy, label, offset, color in [
        ("uniform", "Равномерно", -0.2, "#387F8C"),
        ("margin", "По модулю отступа", 0.2, "#B84452"),
    ]:
        counts, _ = np.histogram(margins[strategy], bins=bins)
        percentages = 100 * counts / len(margins[strategy])
        bars = axis.bar(
            positions + offset, percentages, width=0.4, label=label, color=color
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set(
        title=f"По {len(margins['uniform']):,} предъявлений · веса фиксированы".replace(",", " "),
        xlabel="Модуль отступа выбранного объекта |M|",
        ylabel="Доля предъявлений, %",
        xticks=positions,
        xticklabels=labels,
    )
    axis.margins(y=0.2)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    return figure


def print_results(
    prepared_data: PreparedData,
    results: EvaluationResults,
) -> None:
    print(prepared_data.cleaning_summary.to_string())
    print(prepared_data.distribution.to_string())
    print(prepared_data.split_summary.to_string(index=False))

    print("\nValidation: лучшие старты каждого варианта")
    columns = ["run", "epochs", "stop_reason", "f1_red", "validation_loss"]
    summary = results.validation_summary[columns].copy()
    summary["stop_reason"] = summary["stop_reason"].map({
        "weights_stabilized": "Стабилизация весов",
        "epoch_limit": "Лимит эпох",
    })
    print(summary.to_string(index=False))
    print("Лучший запуск:", results.best_run)
    print("\nКачество на test")
    print(results.test_metrics.to_string(index=False))
    print("\nОтступы после обучения")
    print(results.margin_summary.to_string(index=False))


def create_figures(
    prepared_data: PreparedData,
    results: EvaluationResults,
    *,
    sampling_margins: dict[str, np.ndarray],
) -> dict[str, Figure]:
    comparison_scores = {
        name: scores
        for name, scores in results.test_scores.items()
        if name != "Всегда белое"
    }
    test_labels = prepared_data.datasets["test"].labels
    return {
        "01_classes.png": plot_class_distribution(prepared_data.distribution),
        "02_initial_margins.png": plot_initial_margins(results.initial_margins),
        "03_learning.png": plot_learning_curves(results.learning_histories),
        "04_recurrent_quality.png": plot_recurrent_quality(
            results.learning_histories,
            results.validation_runs,
        ),
        "05_confusion_matrices.png": plot_confusion_matrices(test_labels, comparison_scores),
        "06_roc_pr.png": plot_roc_precision_recall(test_labels, comparison_scores),
        "07_final_margins.png": plot_final_margins(results.final_margins),
        "08_multistart.png": plot_multistart(results.validation_runs, results.best_run),
        "09_weight_stability.png": plot_weight_stability(results.learning_histories),
        "11_sampling.png": plot_sampling_distributions(sampling_margins),
    }


def save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def save_results(
    prepared_data: PreparedData,
    results: EvaluationResults,
    lab_directory: Path,
    *,
    sampling_margins: dict[str, np.ndarray],
) -> None:
    results_directory = lab_directory / "results"
    figure_directory = lab_directory / "images"
    results_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    results.validation_runs.to_csv(results_directory / "validation_runs.csv", index=False)
    results.test_metrics.to_csv(results_directory / "metrics.csv", index=False)

    figures = create_figures(
        prepared_data,
        results,
        sampling_margins=sampling_margins,
    )
    for filename, figure in figures.items():
        save_figure(figure, figure_directory / filename)
