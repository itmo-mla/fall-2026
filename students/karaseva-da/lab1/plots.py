from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"
IMAGE_DIR = ROOT / "image"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
LABELS = {
    "correlation_init": "Корреляционный старт, случайный порядок",
    "margin_sampling": "Корреляционный старт, порядок по отступу",
    "margin_sampling_random_init": "Случайный старт, порядок по отступу",
    "multistart": "Мультистарт (лучший запуск)",
    "fastest_descent": "Скорейший спуск",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def histories(experiments: dict) -> dict[str, dict]:
    runs = experiments["multistart"]["runs"]
    out = {name: experiments[name]["history"] for name in LABELS if name != "multistart"}
    out["multistart"] = runs[experiments["multistart"]["best_run"] - 1]["history"]
    return {name: out[name] for name in LABELS}


def line_plot(series: dict[str, list[float]], title: str, xlabel: str, ylabel: str, name: str, log_y: bool = False, hline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in series.items():
        ax.plot(values, label=label)
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="black", linewidth=1, label="Минимум Q (замкнутая форма)")
    if log_y:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8)
    save(fig, name)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Сначала запусти python linear_classifier.py")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    experiments = result["experiments"]
    best = result["best_model"]
    analysis = result["margin_analysis"]
    optimum = result["optimum_closed_form"]
    q_star = optimum["train_loss"]
    margins = np.asarray(result["test_margins"])
    y = np.asarray(result["test_labels"])
    scores = np.asarray(result["test_scores"])
    hist = histories(experiments)
    y01 = (y == 1).astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    grid = analysis["quantile_grid"]
    ax.plot(grid, analysis["train_random_init_quantiles"], label="Случайный старт")
    ax.plot(grid, analysis["train_correlation_init_quantiles"], label="Корреляционный старт")
    ax.plot(grid, analysis["train_best_quantiles"], label=f"После обучения ({best['name']})")
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.axhline(1, linestyle=":", color="black", linewidth=1)
    ax.set(title="Отступы обучающей выборки по квантилям", xlabel="Доля объектов (по возрастанию отступа)", ylabel="Отступ M = y · (w·x)")
    ax.legend()
    save(fig, "margins_train.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(margins[y == -1], bins=45, alpha=0.65, label="RainTomorrow = No")
    ax.hist(margins[y == 1], bins=45, alpha=0.65, label="RainTomorrow = Yes")
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set(title="Распределение отступов на тестовой выборке", xlabel="Отступ M = y · (w·x)", ylabel="Количество объектов")
    ax.legend()
    save(fig, "margins_test.png")

    sorted_margins = np.sort(margins)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(sorted_margins) + 1), sorted_margins, linewidth=1.0)
    ax.axhline(0, linestyle="--", linewidth=1.5, label="Граница решения M = 0")
    ax.axhline(1, linestyle=":", linewidth=1.2, label="Уровень M = 1")
    ax.set(title="Объекты тестовой выборки по возрастанию отступа", xlabel="Номер объекта после сортировки", ylabel="Отступ M = y · (w·x)")
    ax.legend()
    save(fig, "margins_sorted_test.png")

    line_plot({LABELS[k]: v["loss"] for k, v in hist.items()}, "Функционал качества Q(w) на обучении", "Эпоха / итерация", "Q(w)", "loss_curves.png", hline=q_star)
    line_plot({LABELS[k]: np.maximum(np.asarray(v["loss"]) - q_star, 1e-6) for k, v in hist.items()}, "Невязка Q(w) − Q*", "Эпоха / итерация", "Q(w) − Q*", "optimality_gap.png", log_y=True)
    line_plot({LABELS[k]: v["smooth_loss"] for k, v in hist.items()}, "Рекуррентная оценка функционала", "Эпоха / итерация", "Q̂", "smooth_loss.png")
    line_plot({LABELS[k]: v["accuracy"] for k, v in hist.items()}, "Доля верных ответов на обучении", "Эпоха / итерация", "Accuracy", "train_accuracy.png")
    line_plot({LABELS[k]: v["weight_norm"] for k, v in hist.items()}, "Норма весов без свободного члена", "Эпоха / итерация", "||w||", "weight_norm.png")

    line_plot({f"μ = {k}": v["history"]["loss"] for k, v in result["inertia"].items()}, "Влияние инерции на сходимость SGD", "Эпоха", "Q(w)", "inertia.png", log_y=True, hline=q_star)

    sweep = result["tau_sweep"]
    taus = [row["tau"] for row in sweep]
    positions = np.arange(len(taus))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(positions, [row["test"]["weight_norm"] for row in sweep], marker="o")
    axes[0].set(title="Норма весов при росте L2", xlabel="τ", ylabel="||w||")
    axes[1].plot(positions, [row["validation"]["pr_auc"] for row in sweep], marker="o", label="validation")
    axes[1].plot(positions, [row["test"]["pr_auc"] for row in sweep], marker="o", label="test")
    axes[1].set(title="PR-AUC при росте L2", xlabel="τ", ylabel="PR-AUC")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(positions, [f"{t:g}" for t in taus])
    save(fig, "l2_sweep.png")

    study = experiments["seed_study"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key in zip(axes, ["f1", "pr_auc", "roc_auc"]):
        means = [study[s][key]["mean"] for s in ["random", "margin"]]
        stds = [study[s][key]["std"] for s in ["random", "margin"]]
        ax.bar(["случайный", "по отступу"], means, yerr=stds, capsize=6, color=["tab:blue", "tab:orange"])
        ax.set(title=key, ylim=(min(means) - 0.05, max(means) + 0.05))
    fig.suptitle("Порядок предъявления: среднее ± std по сидам (test)")
    save(fig, "sampling_seeds.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in zip(axes, ["test", "test_tuned_threshold"], ["Порог 0", f"Порог {best['test_tuned_threshold']['threshold']:.2f} (по validation)"]):
        ConfusionMatrixDisplay(np.asarray(best[key]["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(title)
    fig.suptitle(f"Матрица ошибок лучшей модели ({best['name']})")
    save(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y01, scores, ax=ax, name="Собственная модель")
    ax.set_title("ROC-кривая на тестовой выборке")
    save(fig, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y01, scores, ax=ax, name="Собственная модель")
    ax.set_title("Precision–Recall на тестовой выборке")
    save(fig, "precision_recall_curve.png")

    calibration = best["calibration"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(calibration["mean_predicted"], calibration["fraction_positive"], marker="o", label="Модель")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Идеальная калибровка")
    ax.set(title="Калибровка: p = clip((1 + f)/2, 0, 1)", xlabel="Предсказанная вероятность", ylabel="Доля дождливых дней")
    ax.legend()
    save(fig, "calibration.png")

    top = best["top_weights"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = sorted(top.items(), key=lambda item: item[1])
    ax.barh([name for name, _ in ordered], [value for _, value in ordered])
    ax.set(title="Признаки с наибольшими по модулю весами", xlabel="Вес")
    save(fig, "top_weights.png")

    rows = []
    for name in LABELS:
        entry = experiments["multistart"]["runs"][experiments["multistart"]["best_run"] - 1] if name == "multistart" else experiments[name]
        rows.append({"Метод": LABELS[name], **{k: entry["test"][k] for k in METRICS}})
    rows.append({"Метод": f"Лучшая, порог {best['test_tuned_threshold']['threshold']:.2f}", **{k: best["test_tuned_threshold"][k] for k in METRICS}})
    rows.append({"Метод": "Точный минимум Q (Ridge)", **{k: optimum["test"][k] for k in METRICS}})
    rows.append({"Метод": "LogisticRegression (другая потеря)", **{k: result["reference_logistic_regression"]["test"][k] for k in METRICS}})
    rows.append({"Метод": "SGDClassifier (эталон)", **{k: result["reference_sgd_classifier"]["test"][k] for k in METRICS}})
    shown = pd.DataFrame(rows)
    for col in METRICS:
        shown[col] = shown[col].map(lambda value: f"{value:.3f}")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title("Сравнение методов на тестовой выборке")
    save(fig, "metrics_comparison.png")

    dataset_path = ROOT / result["dataset_path"]
    if dataset_path.exists():
        numeric = pd.read_csv(dataset_path).select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)), corr.columns)
        ax.set_title("Корреляционная матрица числовых признаков")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        save(fig, "correlation_matrix.png")

    print(f"Графики сохранены в {IMAGE_DIR}")


if __name__ == "__main__":
    main()