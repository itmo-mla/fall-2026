from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pca import PCAFromSVD


def main():
    output_dir = Path(__file__).resolve().parents[1]
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 1. Загружаем датасет для регрессии
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Обучаем собственный PCA и эталонный sklearn PCA
    pca_custom = PCAFromSVD()
    pca_custom.fit(X_train)

    pca_sklearn = PCA()
    pca_sklearn.fit(X_train)

    # 3. Сравниваем объяснённую дисперсию
    variance_diff = np.max(
        np.abs(
            pca_custom.explained_variance_ratio_
            - pca_sklearn.explained_variance_ratio_
        )
    )

    print("Датасет Diabetes")
    print("Обучающая выборка:", X_train.shape)
    print("Тестовая выборка:", X_test.shape)
    print("Максимальное отличие долей дисперсии:", variance_diff)

    # 4. Определяем эффективную размерность
    k = pca_custom.effective_dimension(epsilon=0.05)
    print("Эффективная размерность (95% дисперсии):", k)

    # 5. Сравниваем восстановление данных
    custom_k = PCAFromSVD(n_components=k).fit(X_train)
    sklearn_k = PCA(n_components=k).fit(X_train)

    X_reconstructed_custom = custom_k.inverse_transform(
        custom_k.transform(X_train)
    )
    X_reconstructed_sklearn = sklearn_k.inverse_transform(
        sklearn_k.transform(X_train)
    )

    print(
        "Максимальное отличие восстановленных данных:",
        np.max(np.abs(X_reconstructed_custom - X_reconstructed_sklearn)),
    )
    print(
        "Относительная ошибка восстановления:",
        custom_k.reconstruction_error(X_train),
    )

    # 6. График накопленной объяснённой дисперсии
    cumulative_variance = np.cumsum(
        pca_custom.explained_variance_ratio_
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        label="Custom PCA",
    )
    plt.axhline(0.95, linestyle="--", label="95% variance")
    plt.axvline(k, linestyle=":", label=f"k = {k}")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA: effective dimensionality")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "explained_variance.png", dpi=150)
    plt.close()

    # 7. Сравниваем линейную регрессию до и после PCA
    model_original = LinearRegression()
    model_original.fit(X_train, y_train)
    pred_original = model_original.predict(X_test)

    X_train_pca = custom_k.transform(X_train)
    X_test_pca = custom_k.transform(X_test)

    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    pred_pca = model_pca.predict(X_test_pca)

    print("\nLinear regression:")
    print(
        "Original:",
        "MSE =", round(mean_squared_error(y_test, pred_original), 4),
        "R2 =", round(r2_score(y_test, pred_original), 4),
    )
    print(
        "After PCA:",
        "MSE =", round(mean_squared_error(y_test, pred_pca), 4),
        "R2 =", round(r2_score(y_test, pred_pca), 4),
    )

    # Сравнение проекций с учётом возможного различия знаков
    Z_custom = custom_k.transform(X_train)
    Z_sklearn = sklearn_k.transform(X_train)

    signs = np.sign(np.sum(Z_custom * Z_sklearn, axis=0))
    signs[signs == 0] = 1

    projection_diff = np.max(
        np.abs(Z_custom * signs - Z_sklearn)
    )
    assert variance_diff < 1e-10
    assert projection_diff < 1e-10
    assert np.max(
        np.abs(X_reconstructed_custom - X_reconstructed_sklearn)
    ) < 1e-10

    print("Проверка эквивалентности PCA: PASSED")
    print("Максимальное отличие проекций:", projection_diff)
        # 8. График собственных значений (scree plot)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(pca_custom.eigenvalues_) + 1),
        pca_custom.eigenvalues_,
        marker="o",
    )
    plt.axvline(k, linestyle="--", label=f"k = {k}")
    plt.xlabel("Number of components")
    plt.ylabel("Eigenvalue")
    plt.title("PCA: scree plot")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "scree_plot.png", dpi=150)
    plt.close()

    # 9. График относительной ошибки восстановления
    reconstruction_errors = 1 - cumulative_variance
    component_counts = np.arange(1, len(reconstruction_errors) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        component_counts,
        reconstruction_errors,
        marker="o",
    )
    plt.axhline(0.05, linestyle="--", label="Error threshold = 0.05")
    plt.axvline(k, linestyle=":", label=f"k = {k}")
    plt.xlabel("Number of components")
    plt.ylabel("Relative reconstruction error")
    plt.title("PCA: reconstruction error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / "reconstruction_error.png", dpi=150)
    plt.close()

    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    metrics = {
        "dataset": "Diabetes",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "original_features": X_train.shape[1],
        "effective_dimension": k,
        "variance_threshold": 0.95,
        "explained_variance": float(cumulative_variance[k - 1]),
        "reconstruction_error": float(custom_k.reconstruction_error(X_train)),
        "max_variance_difference": float(variance_diff),
        "max_reconstruction_difference": float(
            np.max(np.abs(X_reconstructed_custom - X_reconstructed_sklearn))
        ),
        "max_projection_difference": float(projection_diff),
        "regression_original": {
            "mse": float(mean_squared_error(y_test, pred_original)),
            "r2": float(r2_score(y_test, pred_original)),
        },
        "regression_pca": {
            "mse": float(mean_squared_error(y_test, pred_pca)),
            "r2": float(r2_score(y_test, pred_pca)),
        },
    }

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    main()