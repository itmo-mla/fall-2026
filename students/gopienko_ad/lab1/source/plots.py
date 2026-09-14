from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import confusion_matrix


def _prepare_path(output_dir, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def plot_feature_correlations(
    feature_names,
    correlations,
    output_dir="results/plots",
):
    feature_names = np.asarray(feature_names)
    correlations = np.asarray(correlations)
    order = np.argsort(np.abs(correlations))

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[order], correlations[order])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Correlation with target")
    plt.title("Корреляция признаков с классом")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "01_correlations.png"),
        dpi=160,
    )
    plt.close()

def plot_full_correlation_heatmap(
        df,
        output_dir="results/plots",
):
    correlation_matrix = df.corr(
        numeric_only=True
    )

    labels = correlation_matrix.columns.to_numpy()
    values = correlation_matrix.to_numpy()

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        values,
        aspect="auto",
    )

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(len(labels)),
        labels,
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            plt.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.colorbar(image)
    plt.title("Полная матрица корреляций")
    plt.tight_layout()

    plt.savefig(
        _prepare_path(
            output_dir,
            "00_full_correlation_heatmap.png",
        ),
        dpi=160,
    )

    plt.close()


def plot_margins(
    model,
    X,
    y,
    output_dir="results/plots",
):
    margins = model.margin(X, y)
    correct = margins > 0
    incorrect = margins <= 0

    plt.figure(figsize=(10, 6))
    plt.hist(
        margins[correct],
        bins=25,
        alpha=0.7,
        label="M > 0",
    )

    if np.any(incorrect):
        plt.hist(
            margins[incorrect],
            bins=15,
            alpha=0.7,
            label="M <= 0",
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
        label="M = 0",
    )
    plt.xlabel("Margin M = y(w^T x + b)")
    plt.ylabel("Количество объектов")
    plt.title("Распределение отступов")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "02_margins.png"),
        dpi=160,
    )
    plt.close()


def plot_objective(
    models,
    output_dir="results/plots",
):
    plt.figure(figsize=(11, 6))

    for name, model in models.items():
        history = getattr(model, "history", None)
        if not history or len(history.get("objective", [])) == 0:
            continue

        plt.plot(
            history["objective"],
            label=name,
        )

    plt.xlabel("Epoch / iteration")
    plt.ylabel("Q")
    plt.title("Сходимость методов обучения")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "03_objective.png"),
        dpi=160,
    )
    plt.close()


def plot_recurrent_quality(
    model,
    output_dir="results/plots",
):
    plt.figure(figsize=(10, 6))
    plt.plot(
        model.history["objective"],
        label="Полный Q",
    )
    plt.plot(
        model.history["recurrent_objective"],
        label="Рекуррентный Q",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Quality functional")
    plt.title("Рекуррентная оценка функционала")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "04_recurrent_quality.png"),
        dpi=160,
    )
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_dir="results/plots",
):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["-1", "+1"])
    plt.yticks([0, 1], ["-1", "+1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "05_confusion_matrix.png"),
        dpi=160,
    )
    plt.close()


def plot_multistart_objectives(
    seeds,
    objectives,
    best_seed,
    output_dir="results/plots",
):
    plt.figure(figsize=(10, 6))
    plt.plot(seeds, objectives, marker="o")
    plt.axvline(
        best_seed,
        linestyle="--",
        label=f"best seed = {best_seed}",
    )
    plt.xlabel("Seed")
    plt.ylabel("Validation objective")
    plt.title("Мультистарт: качество случайных инициализаций")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        _prepare_path(output_dir, "06_multistart.png"),
        dpi=160,
    )
    plt.close()
