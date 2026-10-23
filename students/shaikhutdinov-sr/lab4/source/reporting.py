from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data import PreparedData
from evaluation import EvaluationResults


COLORS = {"train": "#287D8E", "test": "#B54B59", "threshold": "#CB8925"}
FEATURE_LABELS = {
    "cement": "Цемент",
    "blast_furnace_slag": "Доменный шлак",
    "fly_ash": "Зола-унос",
    "water": "Вода",
    "superplasticizer": "Суперпластификатор",
    "coarse_aggregate": "Крупный заполнитель",
    "fine_aggregate": "Мелкий заполнитель",
    "age": "Возраст бетона",
}


def plot_explained_variance(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float,
    n_components: int,
) -> Figure:
    component_numbers = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative_variance = 100 * np.cumsum(explained_variance_ratio)
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    axis.bar(
        component_numbers, 100 * explained_variance_ratio,
        color=COLORS["train"], alpha=0.8, label="Доля отдельной компоненты",
    )
    axis.plot(
        component_numbers, cumulative_variance, color=COLORS["test"],
        marker="o", label="Накопленная доля",
    )
    axis.axhline(
        100 * variance_threshold, color=COLORS["threshold"], linestyle="--",
        label=f"Порог: {100 * variance_threshold:.0f}%",
    )
    axis.scatter(
        n_components, cumulative_variance[n_components - 1], s=90,
        facecolors="white", edgecolors=COLORS["test"], linewidths=2, zorder=3,
    )
    axis.annotate(
        f"{n_components} компонент: {cumulative_variance[n_components - 1]:.2f}%",
        xy=(n_components, cumulative_variance[n_components - 1]),
        xytext=(12, -38), textcoords="offset points", ha="left",
        arrowprops={"arrowstyle": "-", "color": "#666666"},
    )
    axis.set(
        title="Дисперсия компонент · обучающая выборка",
        xlabel="Номер компоненты", ylabel="Доля дисперсии, %",
        xticks=component_numbers, ylim=(0, 105),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="center right", frameon=False)
    return figure


def draw_reconstruction_curve(
    axis: Axes,
    errors: pd.DataFrame,
    n_components: int,
) -> None:
    for dataset, label in [("train", "Обучающая выборка"), ("test", "Тестовая выборка")]:
        axis.plot(
            errors.index, errors[f"{dataset}_mse"], marker="o",
            color=COLORS[dataset], label=label,
        )
    axis.axvline(
        n_components, color=COLORS["threshold"], linestyle="--",
        label=f"Выбрано {n_components} компонент",
    )
    axis.set(
        title="Ошибка восстановления признаков", xlabel="Количество компонент",
        ylabel="MSE стандартизованных признаков", xticks=errors.index, ylim=(0, None),
    )
    axis.grid(alpha=0.2)
    axis.legend(
        loc="upper right", facecolor="white", edgecolor="none", framealpha=1,
    )


def draw_feature_errors(
    axis: Axes,
    feature_names: list[str],
    feature_rmse: np.ndarray,
    n_components: int,
) -> None:
    labels = [FEATURE_LABELS[name] for name in feature_names]
    feature_errors = pd.Series(feature_rmse, index=labels).sort_values(ascending=False)
    bars = axis.barh(feature_errors.index, feature_errors, color=COLORS["test"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.invert_yaxis()
    axis.set(
        title=f"Восстановление по признакам: {n_components} компонент",
        xlabel="RMSE, в стандартных отклонениях train", ylabel="Признак",
        xlim=(0, 1.3 * feature_errors.max()),
    )
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend([bars], ["Тестовая выборка"], loc="lower right", frameon=False)


def plot_reconstruction(
    errors: pd.DataFrame,
    feature_names: list[str],
    feature_rmse: np.ndarray,
    n_components: int,
) -> Figure:
    figure, (curve_axis, feature_axis) = plt.subplots(
        1, 2, figsize=(12, 5), width_ratios=[1, 1.2], layout="constrained"
    )
    figure.suptitle("Собственная PCA · базис обучен на train")
    draw_reconstruction_curve(curve_axis, errors, n_components)
    draw_feature_errors(feature_axis, feature_names, feature_rmse, n_components)
    return figure


def plot_regression(
    regression_rmse: pd.Series,
    baseline_rmse: float,
    n_components: int,
    variance_threshold: float,
) -> Figure:
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    axis.plot(
        regression_rmse.index, regression_rmse, marker="o",
        color=COLORS["test"], label="Линейная регрессия после PCA",
    )
    axis.axhline(
        baseline_rmse, color="#555555", linestyle="--",
        label=f"Без PCA: 8 признаков · {baseline_rmse:.2f} МПа",
    )
    axis.axvline(
        n_components, color=COLORS["threshold"], linestyle=":",
        label=f"Порог {100 * variance_threshold:.0f}% на train: {n_components} компонент",
    )
    selected_rmse = regression_rmse.loc[n_components]
    axis.scatter(
        n_components, selected_rmse, s=90,
        facecolors="white", edgecolors=COLORS["test"], linewidths=2, zorder=3,
    )
    axis.annotate(
        f"{selected_rmse:.2f} МПа", xy=(n_components, selected_rmse),
        xytext=(12, 18), textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#666666"},
    )
    axis.set(
        title="Предсказание прочности бетона · тестовая выборка",
        xlabel="Количество компонент PCA", ylabel="RMSE, МПа",
        xticks=regression_rmse.index,
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", facecolor="white", edgecolor="none", framealpha=1)
    return figure


def format_report(
    prepared_data: PreparedData,
    results: EvaluationResults,
    variance_threshold: float,
) -> str:
    n_components = results.n_components
    retained_variance = 100 * results.cumulative_variance[n_components - 1]
    previous_variance = (
        100 * results.cumulative_variance[n_components - 2] if n_components > 1 else 0
    )
    worst_feature = int(np.argmax(results.feature_rmse))
    worst_feature_label = FEATURE_LABELS[prepared_data.feature_names[worst_feature]].lower()
    selected_errors = results.errors.loc[n_components]
    selected_regression_rmse = results.regression_rmse.loc[n_components]
    regression_change = 100 * (selected_regression_rmse / results.baseline_rmse - 1)
    comparison_rows = "\n".join(
        f"| {name} | {row['Расхождение']:.2e} |"
        for name, row in results.comparison.iterrows()
    )
    return rf"""# Лабораторная №4. PCA

[Concrete Compressive Strength](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set): {prepared_data.raw_count} измерений прочности бетона, 8 признаков — состав смеси и возраст. Целевая переменная — прочность при сжатии; в PCA она не участвует. Удалены {prepared_data.missing_count} строк с пропусками и {prepared_data.duplicate_count} полных повторов.

Обучающая/тестовая выборки — **{len(prepared_data.train_features)}/{len(prepared_data.test_features)}**, seed 42. Совпадающие векторы признаков объединены в группы и не разделяются между выборками. Признаки стандартизированы по обучающей выборке: масса компонентов и возраст измеряются в разных единицах.

## PCA через SVD

Для центрированной матрицы признаков $X_c=U\Sigma V^\top$ направления компонент — строки $V^\top$. Проекция на первые $m$ компонент и восстановление:

$$
Z=X_cV_m,\qquad \widehat X=ZV_m^\top+\overline X.
$$

$U$ и $V$ содержат левые и правые сингулярные векторы; диагональ $\Sigma$ — сингулярные числа по убыванию. $V_m$ содержит первые $m$ столбцов $V$, а $\overline X$ — обучающее среднее стандартизованных признаков.

Дисперсия компоненты вычисляется как $\lambda_j=\sigma_j^2/(n-1)$, где $\sigma_j$ — сингулярное число, $n$ — число обучающих объектов. Разложение выполнено через `numpy.linalg.svd`.

## Эффективная размерность

Выбираем минимальное число компонент, сохраняющее не менее **{100 * variance_threshold:.0f}% дисперсии** обучающих признаков:

$$
m=\min\left\{{r:\frac{{\sum_{{j=1}}^r\lambda_j}}{{\sum_{{j=1}}^8\lambda_j}}\ge {variance_threshold}\right\}}.
$$

Это критерий сжатия, а не ранг матрицы.

![Спектр и накопленная доля дисперсии](images/01_explained_variance.png)

**Эффективная размерность — {n_components} из 8:** сохранено {retained_variance:.2f}% дисперсии. Первые {n_components - 1} компонент сохраняют только {previous_variance:.2f}%, поэтому выбранный порог ещё не достигнут.

## Восстановление признаков

Ошибку восстановления считаем в стандартизованном пространстве. MSE усредняет квадрат ошибки по объектам и признакам; RMSE отдельного признака выражен в его стандартных отклонениях на обучении.

![Ошибка восстановления в зависимости от числа компонент и по признакам](images/02_reconstruction.png)

При {n_components} компонентах MSE равна **{selected_errors.train_mse:.4f} на обучающей выборке и {selected_errors.test_mse:.4f} на тестовой**. Наибольшая ошибка на тесте у признака «{worst_feature_label}»: RMSE = {results.feature_rmse[worst_feature]:.3f}. Сохранённая доля общей дисперсии не определяет качество регрессии и точность восстановления каждого признака.

## Регрессия после PCA

Для каждого числа компонент обучена обычная линейная регрессия на той же обучающей выборке. RMSE прочности рассчитана на test; базовая модель использует все 8 исходных признаков без PCA.

![Test RMSE линейной регрессии в зависимости от числа компонент PCA](images/03_regression.png)

При {n_components} компонентах сохранено **{retained_variance:.2f}% дисперсии**, но RMSE равна **{selected_regression_rmse:.2f} МПа** против **{results.baseline_rmse:.2f} МПа** без PCA ({regression_change:+.1f}%). Это дополнительная проверка сохранения информации о прочности: число компонент выбрано по порогу {100 * variance_threshold:.0f}% дисперсии train и по test не подбирается.

## Сравнение с sklearn

Эталон — `sklearn.decomposition.PCA(svd_solver="full")` на той же обучающей матрице. Знаки компонент могут различаться, поэтому сравниваем матрицы проекции $V_mV_m^\top$ и восстановленные признаки. Для них взято максимальное расхождение по всем $m=1,\ldots,8$.

| Проверка | Расхождение |
|---|---:|
{comparison_rows}

## Запуск

```bash
python -m pip install -r requirements.txt
python source/main.py
```
"""


def print_results(prepared_data: PreparedData, results: EvaluationResults) -> None:
    print(
        f"Эффективная размерность: {results.n_components} "
        f"из {len(prepared_data.feature_names)}"
    )
    print(results.errors.round(6).to_string())
    print(results.regression_rmse.round(6).to_string())
    print(f"RMSE без PCA: {results.baseline_rmse:.6f} МПа")
    print(results.comparison.to_string())


def save_results(
    prepared_data: PreparedData,
    results: EvaluationResults,
    variance_threshold: float,
    lab_directory: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    image_directory = lab_directory / "images"
    image_directory.mkdir(exist_ok=True)
    figures = {
        "01_explained_variance.png": plot_explained_variance(
            results.explained_variance_ratio, variance_threshold, results.n_components
        ),
        "02_reconstruction.png": plot_reconstruction(
            results.errors, prepared_data.feature_names,
            results.feature_rmse, results.n_components,
        ),
        "03_regression.png": plot_regression(
            results.regression_rmse, results.baseline_rmse,
            results.n_components, variance_threshold,
        ),
    }
    for filename, figure in figures.items():
        figure.savefig(image_directory / filename, dpi=180)
        plt.close(figure)

    report = format_report(prepared_data, results, variance_threshold)
    (lab_directory / "README.md").write_text(report, encoding="utf-8")
