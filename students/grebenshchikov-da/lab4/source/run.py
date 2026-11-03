"""Запуск эксперимента, численная проверка PCA и построение графиков"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error, r2_score

from data import load_data
from pca import effective_dimension, fit_pca, relative_reconstruction_error


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
THRESHOLD = 0.95


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Оценивает свободный член и веса методом наименьших квадратов"""
    # Добавляем столбец единиц для свободного члена
    design = np.column_stack([np.ones(len(X)), X])
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return weights


def predict_linear_regression(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    return design @ weights


def align_component_signs(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Согласует допустимую неоднозначность знака собственных векторов"""
    # Векторы v и -v задают одну ось
    signs = np.sign(np.sum(ours * reference, axis=1))
    signs[signs == 0] = 1.0
    return ours * signs[:, None]


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    r2 = float(r2_score(y_true, prediction))
    return rmse, r2


def plot_variance(ratio: np.ndarray, dimension: int) -> None:
    indexes = np.arange(1, len(ratio) + 1)
    cumulative = np.cumsum(ratio)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.28)
    ax.bar(indexes, ratio, color="#2f7773", label="Доля отдельной компоненты")
    ax.plot(
        indexes,
        cumulative,
        color="#8d739d",
        marker="o",
        linewidth=2,
        label="Накопленная доля",
    )
    ax.axhline(THRESHOLD, color="#48525c", linestyle="--", label="Порог 95%")
    ax.axvline(dimension, color="#43965a", linestyle=":", label=f"m = {dimension}")
    ax.set_xticks(indexes)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Номер главной компоненты")
    ax.set_ylabel("Доля дисперсии")
    ax.set_title("Выбор эффективной размерности")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "explained_variance.png", dpi=160)
    plt.close(fig)


def plot_regression(rmse_values: list[float], dimension: int) -> None:
    indexes = np.arange(1, len(rmse_values) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.28)
    ax.plot(indexes, rmse_values, color="#2f7773", marker="o", linewidth=2)
    ax.scatter(
        [dimension],
        [rmse_values[dimension - 1]],
        color="#8d739d",
        s=70,
        zorder=3,
        label=f"95% дисперсии: m = {dimension}",
    )
    ax.set_xticks(indexes)
    ax.set_xlabel("Число главных компонент")
    ax.set_ylabel("RMSE на тестовой выборке")
    ax.set_title("Качество линейной регрессии после PCA")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "regression_quality.png", dpi=160)
    plt.close(fig)


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    data = load_data()
    model = fit_pca(data.X_train)
    dimension = effective_dimension(model.explained_variance_ratio, THRESHOLD)

    # sklearn используется только для проверки
    reference = SklearnPCA(svd_solver="full").fit(data.X_train)
    aligned_components = align_component_signs(model.components, reference.components_)

    # Проверяем свойства разложения и совпадение с эталоном
    orthogonality_error = float(
        np.max(np.abs(model.components @ model.components.T - np.eye(len(model.components))))
    )
    full_projection = model.transform(data.X_train)
    full_reconstruction_error = float(
        np.max(np.abs(model.inverse_transform(full_projection) - data.X_train))
    )
    component_difference = float(np.max(np.abs(aligned_components - reference.components_)))
    ratio_difference = float(
        np.max(np.abs(model.explained_variance_ratio - reference.explained_variance_ratio_))
    )

    if orthogonality_error > 1e-10 or full_reconstruction_error > 1e-10:
        raise RuntimeError("Собственная реализация PCA не прошла численную проверку")
    if component_difference > 1e-10 or ratio_difference > 1e-12:
        raise RuntimeError("Собственная реализация PCA не совпала с эталоном")

    rmse_values: list[float] = []
    r2_values: list[float] = []
    for count in range(1, data.X_train.shape[1] + 1):
        train_projection = model.transform(data.X_train, count)
        test_projection = model.transform(data.X_test, count)
        weights = fit_linear_regression(train_projection, data.y_train)
        prediction = predict_linear_regression(test_projection, weights)
        rmse, r2 = regression_metrics(data.y_test, prediction)
        rmse_values.append(rmse)
        r2_values.append(r2)

    cumulative_variance = np.cumsum(model.explained_variance_ratio)
    reconstruction_error = relative_reconstruction_error(
        model,
        data.X_train,
        dimension,
    )

    plot_variance(model.explained_variance_ratio, dimension)
    plot_regression(rmse_values, dimension)

    print(f"Объекты: {len(data.y_train) + len(data.y_test)}, признаки: {data.X_train.shape[1]}")
    print(f"Эффективная размерность: {dimension}")
    print(f"Накопленная дисперсия: {cumulative_variance[dimension - 1]:.4f}")
    print(f"Ошибка восстановления: {reconstruction_error:.4f}")
    print(f"Максимальное отличие компонент от sklearn: {component_difference:.2e}")
    print(f"RMSE при m={dimension}: {rmse_values[dimension - 1]:.3f}")
    print(f"RMSE при всех компонентах: {rmse_values[-1]:.3f}")
    print(f"R2 при m={dimension}: {r2_values[dimension - 1]:.3f}")
    print(f"R2 при всех компонентах: {r2_values[-1]:.3f}")


if __name__ == "__main__":
    main()
