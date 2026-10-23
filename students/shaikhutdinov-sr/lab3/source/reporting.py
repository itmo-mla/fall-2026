from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments import ModelComparison
from model import SVM


COLORS = {"benign": "#287D8E", "malignant": "#B54B59", "support": "#CB8925"}
KERNEL_NAMES = {"linear": "Линейное ядро", "rbf": "RBF-ядро"}


def draw_decision_surface(
    axis: Axes,
    classifier: SVM,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> None:
    grid_features = np.column_stack((horizontal.ravel(), vertical.ravel()))
    scores = classifier.decision_function(grid_features).reshape(horizontal.shape)
    axis.contourf(
        horizontal, vertical, scores >= 0, levels=[-0.5, 0.5, 1.5],
        colors=[COLORS["benign"], COLORS["malignant"]], alpha=0.10,
    )
    axis.contour(horizontal, vertical, scores, levels=[0], colors="#333333", linewidths=1.5)
    axis.contour(
        horizontal, vertical, scores, levels=[-1, 1],
        colors="#777777", linestyles="--", linewidths=1,
    )


def draw_training_objects(
    axis: Axes,
    features: np.ndarray,
    labels: np.ndarray,
    support_indices: np.ndarray,
) -> None:
    for label, name, color in [
        (-1, "Доброкачественная", COLORS["benign"]),
        (1, "Злокачественная", COLORS["malignant"]),
    ]:
        objects = features[labels == label]
        axis.scatter(objects[:, 0], objects[:, 1], s=14, color=color, alpha=0.65, label=name)
    support_features = features[support_indices]
    axis.scatter(
        support_features[:, 0], support_features[:, 1], s=43, facecolors="none",
        edgecolors=COLORS["support"], linewidths=0.9, label="Опорные векторы",
    )


def plot_boundaries(
    boundary_models: dict[str, SVM],
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> Figure:
    minimum = training_features[:, :2].min(axis=0) - 0.3
    maximum = training_features[:, :2].max(axis=0) + 0.3
    horizontal, vertical = np.meshgrid(
        np.linspace(minimum[0], maximum[0], 200),
        np.linspace(minimum[1], maximum[1], 200),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained", sharey=True)
    for axis, (kernel, classifier) in zip(axes, boundary_models.items()):
        draw_decision_surface(axis, classifier, horizontal, vertical)
        draw_training_objects(
            axis, training_features, training_labels, classifier.support_indices
        )
        settings = f"C = {classifier.C:g}"
        if kernel == "rbf":
            settings += f", γ = {classifier.gamma:g}"
        axis.set(
            title=f"{KERNEL_NAMES[kernel]} · {settings}",
            xlabel="Средний радиус (стандартизирован)",
            xlim=(minimum[0], maximum[0]), ylim=(minimum[1], maximum[1]),
        )
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Средняя текстура (стандартизирована)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([], [], color="#333333", label="Граница f(x) = 0"),
        Line2D([], [], color="#777777", linestyle="--", label="Уровни f(x) = ±1"),
    ])
    labels.extend(["Граница f(x) = 0", "Уровни f(x) = ±1"])
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Отдельные модели на двух признаках · обучающая выборка")
    return figure


def draw_support_multipliers(
    axis: Axes,
    classifier: SVM,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    margins = labels * classifier.decision_function(features)
    multipliers = classifier.multipliers
    at_zero = multipliers <= classifier.tolerance
    at_limit = multipliers >= classifier.C - classifier.tolerance
    free = ~at_zero & ~at_limit
    for mask, color, label in [
        (at_zero, "#999999", "λ = 0"),
        (free, COLORS["benign"], "0 < λ < C"),
        (at_limit, COLORS["malignant"], "λ = C"),
    ]:
        axis.scatter(
            margins[mask], multipliers[mask] / classifier.C,
            s=24, color=color, alpha=0.65, label=f"{label}: {mask.sum()}",
        )
    axis.axvline(0, color="#999999", linestyle=":", linewidth=1)
    axis.axvline(1, color="#333333", linestyle="--", linewidth=1)
    if classifier.kernel == "linear":
        axis.annotate(
            "M = 1", xy=(1, 0.6), xytext=(0.25, 0.58),
            textcoords="axes fraction", ha="center",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
        )
    settings = f"C = {classifier.C:g}"
    if classifier.kernel == "rbf":
        settings += f", γ = {classifier.gamma:g}"
    axis.set(
        title=f"{KERNEL_NAMES[classifier.kernel]} · {settings}",
        xlabel="Функциональный отступ M = y · f(x)",
        ylim=(-0.06, 1.12),
    )
    axis.legend(loc="upper right", frameon=False)
    axis.grid(alpha=0.2)


def plot_support_vectors(
    results: list[ModelComparison],
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained", sharey=True)
    for axis, result in zip(axes, results):
        draw_support_multipliers(axis, result.own, training_features, training_labels)
    axes[0].set_ylabel("Множитель Лагранжа λ / C")
    figure.suptitle("Решение на всех 30 признаках · обучающая выборка")
    return figure


def format_parameter_table(
    results: list[ModelComparison],
    validation_results: pd.DataFrame,
) -> str:
    rows = []
    for result in results:
        classifier = result.own
        selected = validation_results[
            (validation_results["kernel"] == result.kernel)
            & (validation_results["C"] == classifier.C)
            & (validation_results["gamma"] == classifier.gamma)
        ].iloc[0]
        gamma = "—" if result.kernel == "linear" else f"{classifier.gamma:g}"
        rows.append(
            f"| {KERNEL_NAMES[result.kernel]} | {classifier.C:g} | {gamma} | {selected.f1:.4f} |"
        )
    return "\n".join(rows)


def format_quality_table(metrics: pd.DataFrame) -> str:
    return "\n".join(
        f"| {KERNEL_NAMES[row.kernel]} | {row.model} | {row.errors} | "
        f"{row.accuracy:.4f} | {row.f1:.4f} | {row.recall:.4f} |"
        for row in metrics.itertuples()
    )


def format_verification_table(diagnostics: pd.DataFrame) -> str:
    return "\n".join(
        f"| {KERNEL_NAMES[row.kernel]} | {row.equality_error:.1e} | {row.duality_gap:.1e} | "
        f"{row.max_score_difference:.1e} | {row.prediction_differences} |"
        for row in diagnostics.itertuples()
    )


def write_report(
    results: list[ModelComparison],
    training_labels: np.ndarray,
    test_labels: np.ndarray,
    validation_results: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    lab_directory: Path,
) -> None:
    parameter_table = format_parameter_table(results, validation_results)
    quality_table = format_quality_table(metrics)
    verification_table = format_verification_table(diagnostics)
    own_metrics = metrics[metrics["model"] == "Собственная"].set_index("kernel")
    linear_metrics = own_metrics.loc["linear"]
    rbf_metrics = own_metrics.loc["rbf"]
    f1_difference = rbf_metrics.f1 - linear_metrics.f1
    recall_difference = rbf_metrics.recall - linear_metrics.recall
    prediction_differences = int(diagnostics["prediction_differences"].sum())
    malignant_count = int(
        np.count_nonzero(training_labels == 1) + np.count_nonzero(test_labels == 1)
    )
    report = rf"""# Лабораторная №3. Метод опорных векторов

[Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data): {len(training_labels) + len(test_labels)} объектов, 30 числовых признаков. Злокачественные опухоли — $y=+1$ ({malignant_count}), доброкачественные — $y=-1$ ({len(training_labels) + len(test_labels) - malignant_count}). `id` и пустой столбец исключены. Train/test — **{len(training_labels)}/{len(test_labels)}**, стратификация, seed 42.

## Двойственная задача и ядра

Реализована минимизация по множителям Лагранжа через `scipy.optimize.minimize`, метод SLSQP, с аналитическим градиентом (`ftol=1e-9`, максимум 2000 итераций):

$$
\begin{{gathered}}
\min_{{\lambda}}\;\frac12\sum_{{i,j}}\lambda_i\lambda_j y_i y_j K(x_i,x_j)-\sum_i\lambda_i,\\
0\le\lambda_i\le C,\qquad\sum_i\lambda_i y_i=0.
\end{{gathered}}
$$

Для линейного ядра $K(x,z)=x^\top z$ восстанавливается $w=\sum_i\lambda_i y_i x_i$. Для RBF используется $K(x,z)=e^{{-\gamma\|x-z\|^2}}$. Ответ определяется знаком оценки:

$$
f(x)=\sum_i\lambda_i y_iK(x_i,x)+b.
$$

$C$ задаёт штраф за нарушения отступа; чем больше $\gamma$, тем быстрее убывает RBF-ядро с расстоянием между объектами.

Смещение $b$ — среднее значений по свободным опорным векторам:

$$
b_i=y_i-\sum_j\lambda_j y_j K(x_j,x_i),\qquad 0\lt\lambda_i\lt C.
$$

Если свободных векторов нет, берём середину допустимого интервала $b$. Его задают условия $M_i\ge1$ при $\lambda_i=0$ и $M_i\le1$ при $\lambda_i=C$, где $M_i=y_i f(x_i)$.

$C\in\{{0.1,1,10\}}$, для RBF $\gamma\in\{{0.01,0.1\}}$ выбраны по F1 на validation: 25% train, стратификация, seed 42. При равенстве F1 выбирались меньшие C и γ. Стандартизация обучалась внутри каждого разбиения; итоговые модели переобучены на всех {len(training_labels)} объектах train.

| Ядро | C | γ | Validation F1 |
|---|---:|---:|---:|
{parameter_table}

![Отступы и множители Лагранжа](images/01_support_vectors.png)

Опорными стали **{int(linear_metrics.support_count)} из {len(training_labels)}** объектов у линейной модели ({linear_metrics.support_count / len(training_labels):.1%}) и **{int(rbf_metrics.support_count)}** у RBF ({rbf_metrics.support_count / len(training_labels):.1%}). Свободные множители $0\lt\lambda\lt C$ сосредоточены около $M=1$. Для итоговых моделей опорными считаются объекты с $\lambda\gt10^{{-6}}$.

## Визуализация границы

Для рисунка отдельно обучены модели **только на `radius_mean` и `texture_mean`**. Параметры указаны над панелями, кольцами отмечены опорные векторы. Эти модели не участвуют в таблице качества на 30 признаках.

![Линейная и нелинейная границы на двух признаках](images/02_boundaries.png)

## Сравнение со sklearn SVC

Одинаковые данные, стандартизация, C и γ; у SVC `tol=1e-8`. F1 и Recall рассчитаны для злокачественного класса.

| Ядро | Реализация | Ошибки из {len(test_labels)} | Accuracy | F1 | Recall |
|---|---|---:|---:|---:|---:|
{quality_table}

Линейная модель пропустила **{int(linear_metrics.false_negatives)}** злокачественных случая и дала **{int(linear_metrics.false_positives)}** ложноположительных ответов; RBF — **{int(rbf_metrics.false_negatives)}** и **{int(rbf_metrics.false_positives)}** соответственно. При переходе к RBF Recall изменился на **{recall_difference:+.4f}**, F1 — на **{f1_difference:+.4f}**. Расхождений ответов собственной и библиотечной реализаций для обоих ядер — **{prediction_differences}**.

Невязка — $\lvert\sum_i\lambda_i y_i\rvert$; $\Delta f=f-f_{{SVC}}$ — разность оценок на test.

| Ядро | Невязка | $P-D$ | Макс. $\lvert\Delta f\rvert$ | Разных ответов |
|---|---:|---:|---:|---:|
{verification_table}

Прямой функционал на train:

$$
P=\frac12\|w\|^2+C\sum_i\max(0,1-M_i).
$$

Двойственная цель $D$ — с противоположным знаком к минимизируемому выше выражению. Разрыв $P-D$ в оптимуме равен нулю. Для RBF квадрат нормы вычисляется через ядро: $\|w\|^2=\sum_{{i,j}}\lambda_i\lambda_j y_i y_j K(x_i,x_j)$.

## Запуск

```bash
python -m pip install -r requirements.txt
python source/main.py
```
"""
    (lab_directory / "README.md").write_text(report, encoding="utf-8")


def save_figures(
    results: list[ModelComparison],
    boundary_models: dict[str, SVM],
    training_features: np.ndarray,
    training_labels: np.ndarray,
    lab_directory: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    images_directory = lab_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "01_support_vectors.png": plot_support_vectors(
            results, training_features, training_labels
        ),
        "02_boundaries.png": plot_boundaries(
            boundary_models, training_features, training_labels
        ),
    }
    for filename, figure in figures.items():
        figure.savefig(images_directory / filename, dpi=180)
        plt.close(figure)
