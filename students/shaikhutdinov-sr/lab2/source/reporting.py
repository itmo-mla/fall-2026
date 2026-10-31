from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from experiments import ExperimentResults


COLORS = {
    "Парзен": "#287D8E",
    "sklearn KNN": "#CB8925",
    "Парзен с эталонами": "#B54B59",
}


def plot_loo_curves(results: ExperimentResults) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for result in (results.full, results.reference, results.reduced):
        curve = 100 * result.loo_curve
        best_k = int(curve.idxmin())
        label = f"{result.name}\nМинимум: k = {best_k}, {curve.loc[best_k]:.2f}%"
        axis.plot(curve.index, curve, color=COLORS[result.name], label=label)
        axis.scatter(
            best_k, curve.loc[best_k], s=55, facecolors="white",
            edgecolors=COLORS[result.name], linewidths=1.8, zorder=3,
        )

    axis.set(
        title="Зависимость ошибки LOO от k", xlabel="k", ylabel="Ошибка LOO, %",
        xlim=(0.5, 50.5), ylim=(0, 51), xticks=[1, 10, 20, 30, 40, 50],
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", frameon=False)
    return figure


def plot_prototype_selection(
    results: ExperimentResults,
    y_train: np.ndarray,
) -> Figure:
    history = results.selection_history
    parzen_errors = np.rint(history["parzen_risk"] * len(y_train)).astype(int)
    full_errors = int(round(results.full.loo_curve.loc[1] * len(y_train)))
    counts = pd.Series(y_train[results.prototype_indices]).value_counts().sort_index()
    counts = counts.sort_values(ascending=False, kind="stable")
    figure, (progress_axis, composition_axis) = plt.subplots(
        1, 2, figsize=(12, 6.5), width_ratios=[1.35, 1], layout="constrained"
    )

    progress_axis.plot(
        history["prototypes"], history["errors"], color="#666666",
        marker="o", markersize=3, label="1NN — критерий отбора",
    )
    progress_axis.plot(
        history["prototypes"], parzen_errors, color=COLORS["Парзен с эталонами"],
        marker="o", markersize=3, label="Парзен, k = 1 — оценка набора",
    )
    progress_axis.axhline(
        full_errors, color=COLORS["Парзен"], linestyle="--",
        label=f"Парзен, k = 1, полная база: {full_errors}",
    )
    first = history.iloc[0]
    progress_axis.annotate(
        f"Эталонов: {first.prototypes:.0f}\nОшибок: {first.errors:.0f}",
        xy=(first.prototypes, first.errors), xytext=(12, -5),
        textcoords="offset points", ha="left",
        arrowprops={"arrowstyle": "-", "color": "#777777"},
    )
    for name, errors, offset in [
        ("1NN", history["errors"].iloc[-1], (-12, -14)),
        ("Парзен", parzen_errors.iloc[-1], (-12, 20)),
    ]:
        progress_axis.annotate(
            f"{name}: {errors:.0f}",
            xy=(history["prototypes"].iloc[-1], errors), xytext=offset,
            textcoords="offset points", ha="right",
            arrowprops={"arrowstyle": "-", "color": "#777777"},
        )
    progress_axis.set(
        title="Ошибки 1NN и Парзена при отборе эталонов",
        xlabel="Количество эталонов", ylabel="Число ошибок на обучающей выборке",
        ylim=(0, 1.12 * history["errors"].max()),
    )
    progress_axis.grid(alpha=0.2)
    progress_axis.legend(loc="upper right", frameon=False)

    bars = composition_axis.barh(counts.index, counts, color=COLORS["Парзен с эталонами"])
    composition_axis.bar_label(bars, padding=3)
    composition_axis.invert_yaxis()
    composition_axis.set(
        title="Распределение эталонов по классам",
        xlabel="Количество эталонов", ylabel="Класс",
        xlim=(0, counts.max() + 1),
    )
    composition_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    composition_axis.tick_params(axis="y", labelsize=9)
    composition_axis.grid(axis="x", alpha=0.2)
    composition_axis.set_axisbelow(True)
    composition_axis.legend([bars], ["Число отобранных объектов"], loc="lower right", frameon=False)
    return figure


def plot_class_errors(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> Figure:
    names = list(predictions)
    pairs = sorted({
        (true_class, predicted_class)
        for predicted in predictions.values()
        for true_class, predicted_class in zip(y_test, predicted)
        if true_class != predicted_class
    })
    error_counts = np.array([
        [np.count_nonzero((y_test == true_class) & (predictions[name] == predicted_class))
         for name in names]
        for true_class, predicted_class in pairs
    ])
    figure, axis = plt.subplots(figsize=(8, max(3, 0.38 * len(pairs) + 1.5)), layout="constrained")
    if not pairs:
        axis.text(0.5, 0.5, "Все объекты классифицированы верно", ha="center", va="center")
        axis.set_axis_off()
        return figure

    image = axis.imshow(error_counts, cmap="Oranges", vmin=0, vmax=error_counts.max(), aspect="auto")
    for row, column in np.ndindex(error_counts.shape):
        count = error_counts[row, column]
        axis.text(
            column, row, str(count), ha="center", va="center",
            color="white" if count >= 0.7 * error_counts.max() else "#333333",
        )
    totals = [np.count_nonzero(predictions[name] != y_test) for name in names]
    axis.set(
        title="Ошибки классификации на тестовой выборке",
        xticks=range(len(names)),
        xticklabels=[f"{name}\nОшибок: {total}" for name, total in zip(names, totals)],
        yticks=range(len(pairs)),
        yticklabels=[f"{true_class} → {predicted_class}" for true_class, predicted_class in pairs],
        ylabel="Истинный класс → предсказанный класс",
    )
    axis.tick_params(length=0)
    axis.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    figure.colorbar(image, ax=axis, label="Число ошибок", ticks=range(error_counts.max() + 1))
    return figure


def write_report(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    results: ExperimentResults,
    metrics: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    lab_directory: Path,
) -> None:
    full = metrics.loc["Парзен"]
    reduced = metrics.loc["Парзен с эталонами"]
    prototype_count = len(results.prototype_indices)
    history = results.selection_history
    full_correct = predictions["Парзен"] == y_test
    reduced_correct = predictions["Парзен с эталонами"] == y_test
    corrected = np.count_nonzero(~full_correct & reduced_correct)
    introduced = np.count_nonzero(full_correct & ~reduced_correct)
    disagreements = np.count_nonzero(predictions["Парзен"] != predictions["sklearn KNN"])
    prototype_counts = pd.Series(y_train[results.prototype_indices]).value_counts()

    result_rows = [
        f"| {row.Index} | {row.k:.0f} | {row.neighbor_count:.0f} | {row.errors:.0f} | "
        f"{row.accuracy:.4f} | {row.f1_macro:.4f} | {row.prediction_ms:.2f} |"
        for row in metrics.itertuples()
    ]
    result_table = "\n".join(result_rows)
    report = rf"""# Лабораторная №2. Метрическая классификация

[Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset): {len(y_train) + len(y_test)} объектов, {len(np.unique(y_train))} класса, {x_train.shape[1]} признаков — N, P, K, температура, влажность, pH и осадки. Обучающая/тестовая выборки — **{len(y_train)}/{len(y_test)}**, стратификация, seed 42; параметры стандартизации рассчитаны на обучающей выборке.

## KNN и выбор k

В собственной реализации оценка класса — сумма весов гауссова ядра по объектам этого класса. Расстояние $\rho$ — евклидово; ширина окна $h$ равна расстоянию до $(k+1)$-го соседа:

$$
h(x)=\rho(x,x^{{(k+1)}}),\qquad K(r)=e^{{-2r^2}},\qquad r=\frac{{\rho(x,x_i)}}{{h(x)}}.
$$

Предсказывается класс с максимальной оценкой. Библиотечный `KNeighborsClassifier` использует голосование большинства среди k ближайших соседей.

Выбирали k от 1 до 50 по минимуму LOO-оценки риска — доли ошибок классификации. Проверяемый объект исключается из соседей; масштабирование и набор эталонов при этом фиксированы.

![Подбор k по LOO](images/01_loo_risk.png)

Минимумы: **k = {int(full['k'])} у Парзена**, **k = {int(metrics.loc['sklearn KNN', 'k'])} у sklearn**, **k = {int(reduced['k'])} после отбора**. При увеличении k LOO-риск быстрее растёт для сокращённой базы: при k = 10 он равен {100 * results.reduced.loo_curve.loc[10]:.2f}% против {100 * results.full.loo_curve.loc[10]:.2f}% на полной базе.

## Отбор эталонов

Начальный набор — по одному объекту, ближайшему к среднему вектору своего класса. На каждом шаге добавляли объект с максимальным снижением числа ошибок **1NN**. Отбор завершался, когда ни одно добавление не уменьшало число ошибок. Оценка проводилась на всех {len(y_train)} обучающих объектах с исключением проверяемого объекта из соседей.

На тех же наборах отдельно оценивали **Парзена при фиксированном k = 1**. Эта оценка не участвовала в выборе эталонов.

![Сравнение ошибок 1NN и Парзена при отборе и распределение эталонов по классам](images/02_prototypes.png)

Отобрано **{prototype_count} объектов ({100 * prototype_count / len(y_train):.2f}% обучающей выборки)**. При исходных 80 объектах каждого класса отбор оставил от {prototype_counts.min()} до {prototype_counts.max()} эталонов на класс. Максимум — у `mothbeans` и `pigeonpeas`. Число ошибок 1NN снизилось с {history['errors'].iloc[0]:.0f} до {history['errors'].iloc[-1]:.0f}.

Ошибки 1NN убывают на каждом шаге отбора, а ошибки Парзена могут возрастать. Итоговый набор дал {round(history['parzen_risk'].iloc[-1] * len(y_train))} ошибки Парзена против {round(results.full.loo_curve.loc[1] * len(y_train))} на полной базе.

## Качество на тестовой выборке

| Модель | k | Объектов в базе | Ошибки из {len(y_test)} | Accuracy | Macro F1 | Время, мс |
|---|---:|---:|---:|---:|---:|---:|
{result_table}

Время — медиана семи замеров классификации всей тестовой выборки после прогрева. Macro F1 усреднён по {len(np.unique(y_test))} классам.

Предсказания Парзена на полной базе и sklearn различаются на **{disagreements} объектах**: качество близко, время предсказания у библиотечной реализации меньше. Отбор сократил базу **в {len(y_train) / prototype_count:.1f} раза** и время предсказания Парзена **в {full['prediction_ms'] / reduced['prediction_ms']:.1f} раза**. После отбора {corrected} ранее ошибочных ответа стали верными, а {introduced} верных — ошибочными; итоговое число ошибок выросло с {full['errors']:.0f} до {reduced['errors']:.0f}.

![Ошибки классификации по парам классов](images/03_class_errors.png)

После отбора появились три ошибки `grapes → apple`; число ошибок `rice → jute` выросло с двух до четырёх.

## Запуск

```bash
python -m pip install -r requirements.txt
python source/main.py
```
"""
    (lab_directory / "README.md").write_text(report, encoding="utf-8")


def save_report(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    results: ExperimentResults,
    metrics: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    lab_directory: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    images_directory = lab_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "01_loo_risk.png": plot_loo_curves(results),
        "02_prototypes.png": plot_prototype_selection(results, y_train),
        "03_class_errors.png": plot_class_errors(y_test, predictions),
    }
    for filename, figure in figures.items():
        figure.savefig(images_directory / filename, dpi=180)
        plt.close(figure)
    write_report(x_train, y_train, y_test, results, metrics, predictions, lab_directory)
