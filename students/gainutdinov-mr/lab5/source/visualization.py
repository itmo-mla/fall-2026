"""Визуализация экспериментов лабораторной работы № 5."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(history, y_test, scores, metrics, output_dir, tol):
    """Сохранить графики сходимости IRLS, метрик и ROC обеих моделей.

    history содержит iteration, train_loss и probability_change; для
    начального приближения probability_change равен None. scores содержит
    исходные значения decision_function, metrics — метрики каждой модели.
    Возвращает словарь путей к сохранённым PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "convergence": output_dir / "convergence.png",
        "comparison": output_dir / "comparison.png",
        "roc": output_dir / "roc.png",
    }
    colors = {"IRLS": "#2563eb", "sklearn": "#ea580c"}

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        iterations = [row["iteration"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        changes = [row for row in history if row["probability_change"] is not None]
        change_iterations = [row["iteration"] for row in changes]
        probability_changes = [row["probability_change"] for row in changes]

        for row, scale in enumerate(("linear", "log")):
            loss_axis, change_axis = axes[row]
            scale_name = (
                "обычная шкала" if scale == "linear" else "логарифмическая шкала"
            )
            loss_axis.plot(
                iterations, train_losses, "o-", color=colors["IRLS"], label="IRLS"
            )
            loss_axis.set(
                title=f"Ошибка на обучении — {scale_name}",
                xlabel="Итерация (0 — начальный МНК)",
                ylabel="Средний log-loss",
                yscale=scale,
            )

            values = probability_changes
            if scale == "log":
                values = np.maximum(values, np.finfo(float).tiny)
            change_axis.plot(
                change_iterations,
                values,
                "o-",
                color=colors["IRLS"],
                label="Максимальное изменение вероятности",
            )
            change_axis.axhline(
                tol, color="#475569", linestyle="--", label=f"tol = {tol:g}"
            )
            change_axis.set(
                title=f"Критерий остановки — {scale_name}",
                xlabel="Итерация",
                ylabel="max |σ нового шага − σ прошлого шага|",
                yscale=scale,
            )

        for axis in axes.flat:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        fig.savefig(paths["convergence"], dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        names = list(metrics)
        panels = (
            ("accuracy", "Accuracy на тесте", "Правильные ответы, %", 100.0, ".2f"),
            ("train_loss", "Log-loss на обучении", "Средний log-loss", 1.0, ".3g"),
            ("test_loss", "Log-loss на тесте", "Средний log-loss", 1.0, ".3g"),
            (
                "fit_seconds",
                "Время обучения (один запуск)",
                "Время, мс",
                1000.0,
                ".2f",
            ),
        )
        for axis, (key, title, ylabel, scale, number_format) in zip(
            axes.flat, panels, strict=True
        ):
            values = [metrics[name][key] * scale for name in names]
            for index, (name, value) in enumerate(zip(names, values, strict=True)):
                bars = axis.bar(
                    index, value, width=0.55, color=colors[name], label=name
                )
                axis.bar_label(bars, labels=[format(value, number_format)], padding=4)
            axis.set(
                title=title, ylabel=ylabel, xticks=range(len(names)), xticklabels=names
            )
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.25)
            axis.margins(y=0.2)
            if key == "accuracy":
                axis.set_ylim(0, 110)
            elif key == "train_loss" and all(value > 0 for value in values):
                axis.set_yscale("log")
                axis.set_ylim(min(values) * 0.5, max(values) * 4)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2)
        fig.savefig(paths["comparison"], dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(6.5, 6), layout="constrained")
        for name, model_scores in scores.items():
            fpr, tpr, _ = roc_curve(y_test, model_scores, pos_label=1)
            auc = roc_auc_score(y_test, model_scores)
            axis.plot(
                fpr, tpr, color=colors[name], label=f"{name}: ROC AUC = {auc:.4f}"
            )
        axis.plot(
            [0, 1], [0, 1], "--", color="#64748b", label="Случайный классификатор"
        )
        axis.set(
            title="ROC на тестовой выборке (положительный класс +1)",
            xlabel="Доля ложноположительных ответов (FPR)",
            ylabel="Доля истинноположительных ответов (TPR)",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
        fig.savefig(paths["roc"], dpi=180)
        plt.close(fig)

    return paths
