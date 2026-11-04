from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data import PreparedData
from evaluation import EvaluationResults


COLORS = {"Ньютон–Рафсон": "#287D8E", "IRLS": "#B54B59", "sklearn": "#CB8925"}


def plot_convergence(results: EvaluationResults) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, fit in results.fits.items():
        iterations = np.arange(fit.iterations + 1)
        style = {"color": COLORS[name], "label": name}
        if name == "Ньютон–Рафсон":
            style.update(linestyle="-", linewidth=2.5)
        else:
            style.update(linestyle="none", marker="o", markersize=7, markerfacecolor="none")
        axes[0].plot(iterations, fit.loss_history, **style)
        axes[1].semilogy(iterations, fit.gradient_history, **style)

    reference_loss = results.equivalence.loc["sklearn", "train_loss"]
    axes[0].axhline(
        reference_loss, color=COLORS["sklearn"], linestyle="--", linewidth=1.3,
        label=f"sklearn: {reference_loss:.6f}",
    )
    axes[1].axhline(1e-10, color="#666666", linestyle="--", label="Допуск: 10⁻¹⁰")
    axes[0].set(title="Логистическая потеря на train", ylabel="Средняя logloss")
    axes[1].set(title="Проверка условия остановки", ylabel="Максимальный модуль градиента")
    axes[0].text(
        0.98, 0.67, "Линия Ньютона и точки IRLS совпадают\n"
        f"Макс. расхождение весов методов: {results.trajectory_difference:.2e}",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=10,
    )
    max_iterations = max(fit.iterations for fit in results.fits.values())
    for axis in axes:
        axis.set(xlabel="Итерация (0 — начальные веса)", xticks=np.arange(max_iterations + 1))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    return figure


def draw_precision_recall(axis: Axes, results: EvaluationResults) -> None:
    curve = results.precision_recall
    metrics = results.metrics.loc["Ньютон–Рафсон"]
    positive_fraction = results.confusion[1].sum() / results.confusion.sum()
    axis.step(
        curve.recall, curve.precision, where="post", color=COLORS["Ньютон–Рафсон"],
        label=f"Ньютон–Рафсон · AP = {metrics.average_precision:.3f}",
    )
    axis.axhline(
        positive_fraction, color="#666666", linestyle="--",
        label=f"Доля класса Positive: {positive_fraction:.1%}",
    )
    axis.scatter(
        metrics.recall, metrics.precision, s=65, zorder=3,
        color=COLORS["sklearn"], label="Порог вероятности: 0.5",
    )
    axis.set(
        title="Precision–recall · test",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.02), ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="center left", frameon=False)


def draw_confusion_matrix(axis: Axes, matrix: np.ndarray) -> None:
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for true_class, predicted_class in np.ndindex(matrix.shape):
        count = matrix[true_class, predicted_class]
        axis.text(
            predicted_class, true_class, str(count), ha="center", va="center", fontsize=22,
            color="white" if count > matrix.max() * 0.6 else "#333333",
        )
    axis.set(
        title="Ошибки при пороге 0.5", xlabel="Предсказанный класс", ylabel="Истинный класс",
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
    )
    axis.set_xticks([0.5], minor=True)
    axis.set_yticks([0.5], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="both", length=0)
    axis.figure.colorbar(image, ax=axis, shrink=0.8, label="Число объектов")


def plot_test_results(results: EvaluationResults) -> Figure:
    figure, (curve_axis, matrix_axis) = plt.subplots(
        1, 2, figsize=(12, 5.1), width_ratios=[1.4, 1], layout="constrained",
    )
    draw_precision_recall(curve_axis, results)
    draw_confusion_matrix(matrix_axis, results.confusion)
    return figure


def plot_irls_weights(weight_summary: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    iterations = weight_summary.index
    axis.fill_between(
        iterations, weight_summary.percentile_10, weight_summary.percentile_90,
        color=COLORS["IRLS"], alpha=0.15, label="10–90-й процентили",
    )
    axis.plot(
        iterations, weight_summary["median"], marker="o", color=COLORS["IRLS"],
        linewidth=2, label="Медиана",
    )
    axis.plot(
        iterations, weight_summary.percentile_99, linestyle="--",
        color=COLORS["sklearn"], label="99-й процентиль",
    )
    final_median = weight_summary["median"].iloc[-1]
    axis.annotate(
        f"Медиана: {final_median:.4f}", xy=(iterations[-1], final_median),
        xytext=(-12, -24), textcoords="offset points", ha="right",
        arrowprops={"arrowstyle": "-", "color": "#666666"},
    )
    axis.set(
        title="Перевзвешивание объектов в IRLS · train",
        xlabel="Итерация (0 — до первого обновления)", ylabel=r"Вес объекта $W_{ii}$ (лог. шкала)",
        xticks=iterations, yscale="log",
    )
    axis.set_yticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.25], labels=["0.001", "0.003", "0.01", "0.03", "0.1", "0.25"])
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left", frameon=False)
    return figure


def plot_calibration(calibration: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(7.5, 5.8), layout="constrained")
    errors = np.vstack([
        calibration.observed_fraction - calibration.interval_lower,
        calibration.interval_upper - calibration.observed_fraction,
    ])
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Идеальная калибровка")
    axis.errorbar(
        calibration.mean_probability, calibration.observed_fraction, yerr=errors,
        fmt="o", markersize=7, capsize=5, color=COLORS["Ньютон–Рафсон"],
        label="Positive · 95% интервал Уилсона",
    )
    for group in calibration.itertuples():
        near_right_edge = group.mean_probability > 0.8
        axis.annotate(
            f"n = {group.count}", xy=(group.mean_probability, group.observed_fraction),
            xytext=(-10 if near_right_edge else 10, -16 if near_right_edge else 8),
            textcoords="offset points", ha="right" if near_right_edge else "left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    axis.set(
        title="Калибровка вероятностей · Ньютон–Рафсон · test",
        xlabel="Средняя вероятность класса Positive",
        ylabel="Наблюдаемая доля класса Positive",
        xlim=(-0.03, 1.03), ylim=(-0.03, 1.05),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=False)
    return figure


def format_equivalence_table(comparison: pd.DataFrame) -> str:
    rows = []
    for name, values in comparison.iterrows():
        weight_difference = f"{values.max_weight_difference:.2e}" if name != "sklearn" else "—"
        probability_difference = f"{values.max_probability_difference:.2e}" if name != "sklearn" else "—"
        rows.append(
            f"| {name} | {values.train_loss:.10f} | {values.gradient_norm:.2e} | "
            f"{weight_difference} | {probability_difference} |"
        )
    return "\n".join(rows)


def format_metrics_table(metrics: pd.DataFrame) -> str:
    rows = []
    for name, values in metrics.iterrows():
        rows.append(
            f"| {name} | {values.precision:.4f} | {values.recall:.4f} | "
            f"{values.f1:.4f} | {values.average_precision:.4f} | {values.log_loss:.4f} |"
        )
    return "\n".join(rows)


def write_report(data: PreparedData, results: EvaluationResults, lab_directory: Path) -> None:
    comparison = results.equivalence
    equivalence_table = format_equivalence_table(comparison)
    metrics_table = format_metrics_table(results.metrics)
    stopping = "; ".join(
        f"{name}: {fit.iterations} шагов, "
        + ("остановка по градиенту" if fit.converged else "достигнут лимит итераций")
        for name, fit in results.fits.items()
    )
    _, false_positive, false_negative, true_positive = results.confusion.ravel()
    final_object_weights = results.irls_weights.iloc[-1]
    calibration_group = results.calibration.loc[results.calibration.bin_lower == 0.2].iloc[0]
    report = rf"""# Лабораторная №5. Логистическая регрессия

[HTRU2](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2): 17 898 объектов, 8 числовых признаков: среднее, стандартное отклонение, асимметрия и эксцесс для профиля импульса и кривой DM–SNR. Positive ($y=1$) — пульсары (1 639 объектов), Negative ($y=0$) — остальные объекты (16 259). Данные скачиваются с Kaggle при запуске; пропусков и повторов нет.

Train/test — **{len(data.train.labels)}/{len(data.test.labels)}**, стратификация, seed 42. Признаки стандартизированы по средним и стандартным отклонениям train.

## Методы и сходимость

В матрицу $X$ добавлен столбец единиц; $x_i$ — признаки объекта с этим свободным членом, $w$ — коэффициенты модели, $n$ — размер train. Ответ $y_i\in\{{0,1\}}$ — метка класса, $y$ — вектор ответов. Вероятности класса Positive равны $p=\sigma(Xw)$, где $\sigma(t)=1/(1+e^{{-t}})$ — сигмоида. Средняя логистическая потеря без регуляризации $L$, её градиент $g$ и гессиан $H$:

$$
\begin{{gathered}}
L(w)=\frac1n\sum_i\left[\log(1+e^{{x_i^\top w}})-y_i x_i^\top w\right],\\
g=\frac{{X^\top(p-y)}}n,\qquad H=\frac{{X^\top WX}}n,\\
W=\operatorname{{diag}}(p_i(1-p_i)).
\end{{gathered}}
$$

**Ньютон–Рафсон** решает $H\Delta=-g$ и обновляет $w\leftarrow w+\Delta$. **IRLS** решает взвешенную задачу наименьших квадратов с рабочим ответом $z=Xw+W^{{-1}}(y-p)$:

$$
X^\top WXw_{{new}}=X^\top Wz.
$$

Поскольку $Wz=WXw+y-p$, оба обновления алгебраически совпадают. В коде IRLS сразу вычисляется $Wz$, без деления на нулевые веса; системы решаются через `solve`, без обращения матриц.

Начальные веса нулевые, шаг полный. Остановка: $\|g\|_\infty\le10^{{-10}}$, максимум 100 шагов. {stopping}.

![Изменение logloss и нормы градиента у Ньютона и IRLS](images/01_convergence.png)

Максимальное расхождение коэффициентов Ньютона и IRLS по всем общим итерациям — **{results.trajectory_difference:.2e}**.

## Перевзвешивание объектов в IRLS

Показаны веса $W_{{ii}}=p_i(1-p_i)$ во вспомогательной задаче МНК. Полоса охватывает центральные 80% объектов train; вертикальная шкала логарифмическая.

![Изменение распределения весов объектов по итерациям IRLS](images/03_irls_weights.png)

В начале все веса равны **0.25**. К концу медиана снижается до **{final_object_weights['median']:.4f}**, но 99-й процентиль остаётся **{final_object_weights.percentile_99:.3f}**: у большинства объектов прогнозы близки к 0 или 1, у небольшой части — к 0.5. Это веса МНК, а не ошибки объектов.

## Сравнение с sklearn

Эталон — `LogisticRegression` без регуляризации (`C=np.inf`), с тем же свободным членом и подготовкой данных. Параметры: `solver="lbfgs"`, `tol=1e-12`, `max_iter=1000`. Число итераций L-BFGS — {int(comparison.loc['sklearn', 'iterations'])}; они не равнозначны шагам Ньютона.

$\Delta w$ и $\Delta p$ — разности с эталоном; вероятности сравниваются на test.

| Метод | Train logloss | $\lVert g\rVert_\infty$ | Макс. $\lvert\Delta w\rvert$ | Макс. $\lvert\Delta p\rvert$ |
|---|---:|---:|---:|---:|
{equivalence_table}

Расхождение train logloss с эталоном не превышает **{comparison.loss_difference.max():.2e}**. Близки коэффициенты, вероятности и стационарные значения одной и той же функции.

## Качество на тесте

Порог вероятности — 0.5. Precision, recall и F1 рассчитаны для класса Positive; AP (average precision) — средняя precision, взвешенная по приросту recall.

| Метод | Precision | Recall | F1 | AP | Logloss |
|---|---:|---:|---:|---:|---:|
{metrics_table}

![Precision–recall и матрица ошибок на тестовой выборке](images/02_test_results.png)

Ответы всех трёх моделей совпали. Верно распознано **{true_positive} из {false_negative + true_positive}** объектов Positive; **{false_negative}** ложноотрицательных и **{false_positive}** ложноположительных ответов. Кривая показывает компромисс между recall и precision при изменении порога; матрица — результат при пороге 0.5.

## Калибровка вероятностей

Вероятности test разделены на пять интервалов шириной 0.2. Каждая точка сравнивает средний прогноз с фактической долей класса Positive в группе; $n$ — число объектов, вертикальные отрезки — 95% интервалы Уилсона для этой доли.

![Предсказанные вероятности и наблюдаемые частоты класса Positive](images/04_calibration.png)

В группе 0.2–0.4 средний прогноз — **{calibration_group.mean_probability:.0%}**, фактическая доля — **{calibration_group.observed_fraction:.0%}** ({int(calibration_group.positives)} из {int(calibration_group['count'])}). В средних диапазонах мало объектов и интервалы широкие, поэтому оценка калибровки здесь неточна.

## Запуск

```bash
python -m pip install -r requirements.txt
python source/main.py
```
"""
    (lab_directory / "README.md").write_text(report, encoding="utf-8")


def save_report(data: PreparedData, results: EvaluationResults, lab_directory: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    image_directory = lab_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "01_convergence.png": plot_convergence(results),
        "02_test_results.png": plot_test_results(results),
        "03_irls_weights.png": plot_irls_weights(results.irls_weights),
        "04_calibration.png": plot_calibration(results.calibration),
    }
    for name, figure in figures.items():
        figure.savefig(image_directory / name, dpi=160)
        plt.close(figure)
    write_report(data, results, lab_directory)
