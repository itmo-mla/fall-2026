"""Эксперимент с PCA. Подготовка и центрирование данных - в data.py."""

from pathlib import Path

import numpy as np
from data import load_data
from pca import PCA
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from visualization import plot_predictions, plot_reconstruction_error

OUTPUT_DIR = Path(__file__).with_name("figures")
RANDOM_STATE = 42
MAX_RECONSTRUCTION_ERROR = 0.05
TAU = 0.1


def compare_models(X_train, X_test, y_train, y_test, pca, spectrum):
    reference = SklearnPCA(n_components=pca.n_components, svd_solver="full")
    reference_train = reference.fit_transform(X_train)
    reference_test = reference.transform(X_test)
    G_train, G_test = pca.transform(X_train), pca.transform(X_test)
    reconstruction_difference = np.max(np.abs(
        pca.inverse_transform(G_train) - reference.inverse_transform(reference_train)
    ))
    print(f"PCA: макс. разница восстановлений со sklearn = {reconstruction_difference:.3e}")

    y_mean = y_train.mean()
    weights = spectrum.U @ (
        spectrum.D / (spectrum.D**2 + TAU) * (spectrum.V.T @ (y_train - y_mean))
    )
    ridge_predictions = X_test @ weights + y_mean
    ridge_reference = Ridge(alpha=TAU, solver="svd").fit(X_train, y_train).predict(X_test)
    prediction_difference = np.max(np.abs(ridge_predictions - ridge_reference))
    print(f"Ridge: макс. разница предсказаний со sklearn = {prediction_difference:.3e}")

    models = (
        ("PCA + регрессия", LinearRegression().fit(G_train, y_train).predict(G_test)),
        ("sklearn PCA", LinearRegression().fit(reference_train, y_train).predict(reference_test)),
        (f"Ridge (tau={TAU})", ridge_predictions),
        ("sklearn Ridge", ridge_reference),
    )
    print(f"{'Модель':<18}\t{'RMSE':>12}\t{'R^2':>10}")
    for name, predictions in models:
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        # R^2 показывает, насколько модель лучше предсказания средним: 1 — идеально,
        # 0 — как среднее, меньше 0 — хуже. Так качество проще оценить без масштаба продаж.
        r2 = r2_score(y_test, predictions)
        print(f"{name:<18}\t{rmse:>12.4f}\t{r2:>10.6f}")
    plot_predictions(y_test, models, OUTPUT_DIR / "model_predictions.png")


def main():
    # 1. Датасет
    X_train, X_test, y_train, y_test = load_data(random_state=RANDOM_STATE)

    # Сначала изучаем весь спектр только обучающей матрицы.
    # 2. Реализация PCA через сингулярное разложение
    spectrum = PCA().fit(X_train)
    # 3. Эффективная размерность выборки
    errors = spectrum.calculate_em()
    effective_m = int(np.flatnonzero(errors <= MAX_RECONSTRUCTION_ERROR)[0] + 1)
    plot_reconstruction_error(errors, OUTPUT_DIR / "pca_reconstruction_error.png")

    # Число компонент - настройка объекта, а не глобальная переменная класса.
    pca = PCA(n_components=effective_m)
    G_train = pca.fit_transform(X_train)
    G_test = pca.transform(X_test)
    F_restored = pca.inverse_transform(G_train)
    train_error = (
        np.linalg.norm(X_train - F_restored) ** 2 / np.linalg.norm(X_train) ** 2
    )

    print(
        f"Эффективная размерность при E_m <= {MAX_RECONSTRUCTION_ERROR}: {effective_m}"
    )
    print(f"E_m = {train_error:.8f}")
    print(f"Новые признаки: train {G_train.shape}, test {G_test.shape}")
    # 4. Сравнение с эталонной реализацией
    compare_models(X_train, X_test, y_train, y_test, pca, spectrum)


if __name__ == "__main__":
    main()
