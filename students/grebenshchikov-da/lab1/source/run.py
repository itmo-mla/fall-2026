from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data import load_prepared_data
from linear_classifier import (
    LinearClassifier,
    correlation_initialization,
    empirical_risk,
    fit_multistart,
    margins,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_DIR / "images"
RANDOM_STATE = 42
L2 = 0.01
COLORS = {
    "blue": "#2A6F6B",
    "orange": "#8B6F9C",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#475569",
    "grid": "#CBD5E1",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def fit_base_model(data, learning_rate: float) -> LinearClassifier:
    initial_weights = correlation_initialization(data.X_train, data.y_train)
    return LinearClassifier(l2=L2).fit_sgd(
        data.X_train,
        data.y_train,
        initial_weights=initial_weights,
        epochs=80,
        learning_rate=learning_rate,
        momentum=0.9,
        quality_rate=0.02,
        presentation="random",
        random_state=RANDOM_STATE,
    )


def select_learning_rate(data):
    candidates = [0.0001, 0.0002, 0.0005]
    rows = {}
    models = {}

    for learning_rate in candidates:
        model = fit_base_model(data, learning_rate)
        prediction = model.predict(data.X_validation)
        row = calculate_metrics(data.y_validation, prediction)
        row["train_risk"] = empirical_risk(
            data.X_train,
            data.y_train,
            model.weights,
            L2,
        )
        rows[learning_rate] = row
        models[learning_rate] = model

    metrics = pd.DataFrame.from_dict(rows, orient="index")
    metrics.index.name = "learning_rate"
    best_rate = max(
        candidates,
        key=lambda rate: (
            rows[rate]["balanced_accuracy"],
            rows[rate]["f1"],
            -rows[rate]["train_risk"],
        ),
    )
    return best_rate, metrics, models[best_rate]


def train_models(data, learning_rate: float, base_model: LinearClassifier):
    common = {
        "epochs": 80,
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "quality_rate": 0.02,
    }
    correlation_weights = correlation_initialization(data.X_train, data.y_train)

    models = {
        "Корреляция + случайный порядок": base_model,
        "Мультистарт + случайный порядок": fit_multistart(
            data.X_train,
            data.y_train,
            starts=5,
            presentation="random",
            l2=L2,
            random_state=RANDOM_STATE,
            **common,
        ),
        "Мультистарт + порядок по отступу": fit_multistart(
            data.X_train,
            data.y_train,
            starts=5,
            presentation="margin",
            l2=L2,
            random_state=RANDOM_STATE,
            **common,
        ),
        "Скорейший градиентный спуск": LinearClassifier(l2=L2).fit_steepest(
            data.X_train,
            data.y_train,
            initial_weights=correlation_weights,
        ),
    }

    # sklearn минимизирует сумму квадратов, а наша запись использует среднее.
    # Поэтому alpha = n * lambda / 2 задаёт тот же штраф L2.
    reference = RidgeClassifier(alpha=len(data.y_train) * L2 / 2.0)
    reference.fit(data.X_train, data.y_train)
    return models, reference


def plot_class_balance(data) -> None:
    all_targets = np.concatenate(
        [data.y_train, data.y_validation, data.y_test]
    )
    counts = [np.sum(all_targets == -1), np.sum(all_targets == 1)]

    total = sum(counts)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(
        ["Petrol (-1)", "Diesel (+1)"],
        counts,
        color=[COLORS["blue"], COLORS["orange"]],
        width=0.58,
        zorder=3,
    )
    labels = [f"{count}\n{count / total:.1%}" for count in counts]
    value_labels = ax.bar_label(bars, labels=labels, padding=5, fontsize=11)
    for label in value_labels:
        label.set_zorder(4)
    ax.set_title("Распределение целевых классов")
    ax.set_ylabel("Количество автомобилей")
    ax.set_ylim(0, max(counts) * 1.16)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "class_balance.png", dpi=180)
    plt.close(fig)


def plot_training(model: LinearClassifier) -> None:
    epochs = np.arange(len(model.history["risk"]))
    risk = np.asarray(model.history["risk"])
    quality = np.asarray(model.history["quality"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.plot(
            epochs,
            risk,
            label="Эмпирический риск",
            color=COLORS["blue"],
            linewidth=2.8,
            zorder=3,
        )
        ax.plot(
            epochs,
            quality,
            label="Рекуррентная оценка",
            color=COLORS["orange"],
            linewidth=1.7,
            alpha=0.72,
            zorder=2,
        )
        ax.set_xlabel("Эпоха")

    axes[0].set_title("Весь процесс обучения")
    axes[0].set_ylabel("Значение функционала")
    axes[0].set_yscale("log")
    axes[0].annotate(
        f"старт {risk[0]:.2f}",
        xy=(0, risk[0]),
        xytext=(8, -8),
        textcoords="offset points",
        color=COLORS["slate"],
    )

    axes[1].set_title("После первой эпохи")
    axes[1].set_xlim(1, epochs[-1] + 1.5)
    zoom_values = np.concatenate([risk[1:], quality[1:]])
    padding = max((zoom_values.max() - zoom_values.min()) * 0.12, 0.002)
    axes[1].set_ylim(zoom_values.min() - padding, zoom_values.max() + padding)
    axes[1].scatter(
        epochs[-1],
        risk[-1],
        s=38,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    axes[1].text(
        0.97,
        0.92,
        f"Итоговый риск\n{risk[-1]:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color="#0F172A",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
            "alpha": 0.95,
        },
    )

    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Сходимость SGD с корреляционной инициализацией", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMAGES_DIR / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multistart(models: dict[str, LinearClassifier]) -> None:
    variants = [
        (
            "Мультистарт + случайный порядок",
            "Случайный порядок",
            COLORS["blue"],
            "o",
        ),
        (
            "Мультистарт + порядок по отступу",
            "Порядок по |M|",
            COLORS["purple"],
            "s",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (model_name, label, color, marker) in zip(axes, variants):
        model = models[model_name]
        risks = np.asarray(model.multistart_risks)
        starts = np.arange(1, len(risks) + 1)
        ax.plot(
            starts,
            risks,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=1.8,
            label=label,
        )
        best_index = int(np.argmin(risks))
        for index, (start, risk) in enumerate(zip(starts, risks)):
            if index == best_index:
                continue
            ax.annotate(
                f"{risk:.5f}",
                xy=(start, risk),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color="#0F172A",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": COLORS["grid"],
                    "alpha": 0.9,
                },
            )

        ax.scatter(
            starts[best_index],
            risks[best_index],
            marker="o",
            s=95,
            color=COLORS["green"],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.annotate(
            f"Лучший №{best_index + 1}\n{risks[best_index]:.5f}",
            xy=(starts[best_index], risks[best_index]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#0F172A",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
                "alpha": 0.9,
            },
        )
        local_padding = max((risks.max() - risks.min()) * 0.45, 0.0006)
        ax.set_ylim(risks.min() - local_padding, risks.max() + local_padding)
        ax.set_xlim(0.65, len(starts) + 0.35)
        ax.set_xticks(starts)
        ax.set_xlabel("Номер инициализации")
        ax.set_title(label)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Функционал на обучающей выборке")
    fig.suptitle("Влияние начальных весов на результат мультистарта", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(IMAGES_DIR / "multistart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_margins(data, model: LinearClassifier) -> None:
    before = np.sort(margins(data.X_train, data.y_train, model.initial_weights))
    after = np.sort(margins(data.X_train, data.y_train, model.weights))

    lower = min(float(before.min()), float(after.min()), 0.0)
    upper = max(float(before.max()), float(after.max()), 1.0)
    padding = max((upper - lower) * 0.08, 0.08)
    lower_limit = lower - padding
    upper_limit = upper + padding

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    for ax, values, title, color in [
        (axes[0], before, "До обучения", COLORS["slate"]),
        (axes[1], after, "После обучения", COLORS["blue"]),
    ]:
        percentile = np.linspace(0, 100, len(values))
        ax.axhspan(lower_limit, 0, color=COLORS["red"], alpha=0.13)
        ax.axhspan(0, 1, color=COLORS["orange"], alpha=0.13)
        ax.axhspan(1, upper_limit, color=COLORS["green"], alpha=0.11)
        ax.plot(percentile, values, color=color, linewidth=2.4, zorder=3)
        ax.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1.5)
        ax.axhline(1, color=COLORS["green"], linestyle=":", linewidth=1.8)

        error_count = int(np.sum(values < 0))
        below_target = int(np.sum((values >= 0) & (values < 1)))
        target_reached = int(np.sum(values >= 1))
        ax.text(
            0.03,
            0.95,
            (
                f"M < 0:  {error_count}\n"
                f"0 ≤ M < 1:  {below_target}\n"
                f"M ≥ 1:  {target_reached}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.94,
                "edgecolor": COLORS["grid"],
            },
        )
        ax.set_title(title)
        ax.set_xlabel("Объекты, отсортированные по отступу, %")
        ax.set_xlim(0, 100)
        ax.set_ylim(lower_limit, upper_limit)

    axes[0].set_ylabel("Отступ M = y f(x)")
    legend = [
        Patch(facecolor=COLORS["red"], alpha=0.22, label="M < 0, ошибка"),
        Patch(facecolor=COLORS["orange"], alpha=0.22, label="0 ≤ M < 1"),
        Patch(facecolor=COLORS["green"], alpha=0.20, label="M ≥ 1"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Как обучение изменило распределение отступов", y=0.99)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(IMAGES_DIR / "margins.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, best_name: str) -> None:
    columns = ["accuracy", "balanced_accuracy", "f1"]
    labels = {
        "Корреляция + случайный порядок": "Корреляция + случайно",
        "Мультистарт + случайный порядок": "Мультистарт + случайно",
        "Мультистарт + порядок по отступу": "Мультистарт + |M|",
        "Скорейший градиентный спуск": "Скорейший спуск",
        "RidgeClassifier": "RidgeClassifier",
    }
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "f1": "F1",
    }
    metric_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    y_positions = np.arange(len(metrics))
    offsets = [-0.20, 0.0, 0.20]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    best_position = list(metrics.index).index(best_name)
    ax.axhspan(best_position - 0.42, best_position + 0.42, color=COLORS["blue"], alpha=0.07)

    for column, label, color, offset in zip(columns, metric_labels.values(), metric_colors, offsets):
        values = metrics[column].to_numpy()
        ax.scatter(values, y_positions + offset, s=65, color=color, label=label, zorder=3)
        for value, y in zip(values, y_positions + offset):
            ax.text(value + 0.00055, y, f"{value:.4f}", va="center", fontsize=8.5, color=color)

    minimum = float(metrics[columns].min().min())
    lower_bound = max(0.0, np.floor((minimum - 0.005) * 100) / 100)
    ax.set_xlim(lower_bound, 1.006)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([labels[name] for name in metrics.index])
    ax.invert_yaxis()
    ax.set_title("Качество на тестовой выборке")
    ax.set_xlabel("Значение метрики, увеличенный масштаб")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_true, custom_pred, reference_pred) -> None:
    matrices = [
        confusion_matrix(y_true, custom_pred, labels=[-1, 1]),
        confusion_matrix(y_true, reference_pred, labels=[-1, 1]),
    ]
    titles = ["Лучшая собственная модель", "RidgeClassifier"]
    color_maps = [
        LinearSegmentedColormap.from_list(
            "teal_matrix",
            ["#F8FAFC", COLORS["blue"]],
        ),
        LinearSegmentedColormap.from_list(
            "purple_matrix",
            ["#F8FAFC", COLORS["orange"]],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, matrix, title, cmap in zip(axes, matrices, titles, color_maps):
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(item.max() for item in matrices))
        row_totals = matrix.sum(axis=1, keepdims=True)
        for row in range(2):
            for column in range(2):
                percentage = matrix[row, column] / row_totals[row, 0]
                text_color = "white" if matrix[row, column] > matrix.max() * 0.55 else "#0F172A"
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n{percentage:.1%}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )
        ax.set_title(title)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_xticks([0, 1], ["Petrol", "Diesel"])
        ax.set_yticks([0, 1], ["Petrol", "Diesel"])
        ax.grid(False)

    fig.suptitle("Матрицы ошибок на тестовой выборке", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_signal(data) -> None:
    frame = data.train_frame
    petrol = frame[frame["target"] == -1]
    diesel = frame[frame["target"] == 1]
    specifications = [
        ("torque_rpm", "Обороты максимального момента", "об/мин"),
        ("torque_nm", "Максимальный крутящий момент", "Н·м"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (column, title, unit) in zip(axes, specifications):
        values = [petrol[column].to_numpy(), diesel[column].to_numpy()]
        box = ax.boxplot(
            values,
            tick_labels=["Petrol", "Diesel"],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2.2},
        )
        for patch, color in zip(box["boxes"], [COLORS["blue"], COLORS["orange"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        medians = [np.median(value) for value in values]
        for position, median in enumerate(medians, start=1):
            ax.annotate(
                f"{median:.0f}",
                xy=(position, median),
                xytext=(20, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#CBD5E1",
                    "alpha": 0.9,
                },
            )
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Обоснование разделения классов", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(IMAGES_DIR / "feature_signal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    data = load_prepared_data(PROJECT_DIR, random_state=RANDOM_STATE)
    learning_rate, learning_rate_metrics, base_model = select_learning_rate(data)
    models, reference = train_models(data, learning_rate, base_model)

    validation_rows = {}
    for name, model in models.items():
        prediction = model.predict(data.X_validation)
        row = calculate_metrics(data.y_validation, prediction)
        row["train_risk"] = empirical_risk(
            data.X_train,
            data.y_train,
            model.weights,
            L2,
        )
        validation_rows[name] = row
    validation_metrics = pd.DataFrame.from_dict(validation_rows, orient="index")

    best_name = max(
        models,
        key=lambda name: (
            validation_rows[name]["balanced_accuracy"],
            validation_rows[name]["f1"],
            -validation_rows[name]["train_risk"],
        ),
    )
    best_model = models[best_name]

    rows = {}
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(data.X_test)
        predictions[name] = prediction
        rows[name] = calculate_metrics(data.y_test, prediction)

    reference_name = "RidgeClassifier"
    predictions[reference_name] = reference.predict(data.X_test)
    rows[reference_name] = calculate_metrics(data.y_test, predictions[reference_name])
    metrics = pd.DataFrame.from_dict(rows, orient="index")

    plot_class_balance(data)
    plot_feature_signal(data)
    plot_training(models["Корреляция + случайный порядок"])
    plot_multistart(models)
    plot_margins(data, best_model)
    plot_metric_comparison(metrics, best_name)
    plot_confusion_matrices(
        data.y_test,
        predictions[best_name],
        predictions[reference_name],
    )

    print("Очистка данных")
    for message in data.cleaning_log:
        print("-", message)
    print("\nПодбор шага на валидационной выборке")
    print(learning_rate_metrics.round(4).to_string())
    print(f"\nВыбранный шаг: {learning_rate}")
    print("\nСравнение собственных методов на валидационной выборке")
    print(validation_metrics.round(4).to_string())
    print(f"\nЛучшая собственная модель по валидации: {best_name}")
    print("\nИтоговые метрики на тестовой выборке")
    print(metrics.round(4).to_string())
    print(f"Графики сохранены в {IMAGES_DIR}")


if __name__ == "__main__":
    main()
