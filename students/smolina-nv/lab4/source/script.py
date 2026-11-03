import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
PLOTS_DIR = "plots"
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)




def load_data():
    df = pd.read_csv(DATA_URL, sep=";")

    if "quality" not in df.columns:
        raise ValueError("В датасете отсутствует целевая переменная quality")
    if df.isna().any().any():
        raise ValueError("В датасете обнаружены пропущенные значения")

    feature_names = df.drop(columns="quality").columns.tolist()
    X = df[feature_names].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)
    return X, y, feature_names


def standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    if np.any(std == 0):
        raise ValueError("В данных есть признак с нулевым стандартным отклонением")
    return {"mean": mean, "std": std}


def standardize_transform(X, scaler):
    return (np.asarray(X, dtype=float) - scaler["mean"]) / scaler["std"]



def fit_pca_svd(X, n_components=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X должна быть двумерной матрицей с минимум двумя объектами")

    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    if (not isinstance(n_components, (int, np.integer))
            or not 1 <= n_components <= max_components):
        raise ValueError(f"n_components должно быть целым числом от 1 до {max_components}")

    mean = X.mean(axis=0)
    X_centered = X - mean


    _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)

    all_variances = singular_values ** 2 / (n_samples - 1)
    total_variance = all_variances.sum()
    if total_variance <= 0:
        raise ValueError("Общая дисперсия должна быть положительной")

    return {
        "mean": mean,
        "components": Vt[:n_components],
        "singular_values": singular_values[:n_components],
        "explained_variance": all_variances[:n_components],
        "explained_variance_ratio": (all_variances / total_variance)[:n_components],
        "n_components": n_components,
        "n_features": n_features,
    }


def transform_pca(X, model):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != model["n_features"]:
        raise ValueError("Размерность X не соответствует обученной PCA")
    return (X - model["mean"]) @ model["components"].T


def fit_transform_pca(X, n_components=None):
    model = fit_pca_svd(X, n_components)
    return model, transform_pca(X, model)


def inverse_transform_pca(Z, model):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != model["n_components"]:
        raise ValueError("Неверное количество главных компонент в Z")
    return Z @ model["components"] + model["mean"]



def determine_effective_dimension(model, thresholds=THRESHOLDS):
    if model["n_components"] != model["n_features"]:
        raise ValueError("Для эффективной размерности нужен PCA со всеми компонентами")

    ratios = model["explained_variance_ratio"]
    cumulative = np.cumsum(ratios)
    residual = np.maximum(0.0, 1.0 - cumulative)
    results = {}

    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("Порог должен принадлежать (0, 1]")
        matching = np.flatnonzero(cumulative >= threshold - 1e-12)
        m = int(matching[0] + 1) if matching.size else len(cumulative)
        results[threshold] = {
            "dimension": m,
            "retained_variance": cumulative[m - 1],
            "residual_error": residual[m - 1],
            "reduction_percent": 100 * (1 - m / len(ratios)),
        }
    return results, cumulative, residual


def print_pca_results(model, results, cumulative):
    print("\n1. PCA ЧЕРЕЗ SVD")
    print(f"{'PC':<7}{'Собств. значение':>20}{'Доля':>14}{'Накопленная':>17}")
    print("-" * 58)
    for i, (value, ratio, acc) in enumerate(zip(
        model["explained_variance"], model["explained_variance_ratio"], cumulative
    ), start=1):
        print(f"PC{i:<5}{value:>20.8f}{ratio:>14.8f}{acc:>17.8f}")

    print("\n2. ЭФФЕКТИВНАЯ РАЗМЕРНОСТЬ")
    print(f"{'Порог':<9}{'m':>5}{'Сохранено':>15}{'Потеря E_m':>17}{'Сокращение':>15}")
    print("-" * 61)
    for threshold, row in results.items():
        print(
            f"{threshold:<9.0%}{row['dimension']:>5}"
            f"{row['retained_variance']:>15.6%}"
            f"{row['residual_error']:>17.6%}"
            f"{row['reduction_percent']:>14.1f}%"
        )


def plot_explained_variance(model, results, cumulative):
    positions = np.arange(1, len(cumulative) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    bars = ax1.bar(positions, model["explained_variance_ratio"],
                   label="Доля дисперсии компоненты")
    for bar, value in zip(bars, model["explained_variance_ratio"]):
        ax1.annotate(f"{value:.3f}",
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax1.set(title="Scree plot", xlabel="Номер главной компоненты",
            ylabel="Доля объяснённой дисперсии")
    ax1.set_xticks(positions)
    ax1.set_ylim(0, model["explained_variance_ratio"].max() * 1.18)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.plot(positions, cumulative, "o-", label="Накопленная доля дисперсии")
    for threshold, row in results.items():
        ax2.axhline(threshold, linestyle="--", alpha=0.75,
                    label=f"Порог {threshold:.0%}: {row['dimension']} компонент")
        ax2.axvline(row["dimension"], linestyle=":", alpha=0.3)
    ax2.set(title="Эффективная размерность", xlabel="Количество главных компонент",
            ylabel="Накопленная объяснённая дисперсия")
    ax2.set_xticks(positions)
    ax2.set_xlim(0.5, len(cumulative) + 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return fig


def compare_with_sklearn(X, n_components):
    own, Z_own = fit_transform_pca(X, n_components)
    reference = SklearnPCA(n_components=n_components, svd_solver="full")
    Z_ref = reference.fit_transform(X)

    signs = np.where(np.sum(own["components"] * reference.components_, axis=1) < 0,
                     -1.0, 1.0)
    aligned_components = own["components"] * signs[:, None]
    aligned_scores = Z_own * signs

    own_reconstructed = inverse_transform_pca(Z_own, own)
    ref_reconstructed = reference.inverse_transform(Z_ref)

    metrics = {
        "variance_max_diff": np.max(np.abs(own["explained_variance"] - reference.explained_variance_)),
        "ratio_max_diff": np.max(np.abs(own["explained_variance_ratio"] - reference.explained_variance_ratio_)),
        "singular_max_diff": np.max(np.abs(own["singular_values"] - reference.singular_values_)),
        "components_max_diff": np.max(np.abs(aligned_components - reference.components_)),
        "scores_mae": np.mean(np.abs(aligned_scores - Z_ref)),
        "scores_rmse": np.sqrt(np.mean((aligned_scores - Z_ref) ** 2)),
        "scores_max_diff": np.max(np.abs(aligned_scores - Z_ref)),
        "mse_own": np.mean((X - own_reconstructed) ** 2),
        "mse_ref": np.mean((X - ref_reconstructed) ** 2),
        "reconstruction_max_diff": np.max(np.abs(own_reconstructed - ref_reconstructed)),
        "orthogonality_own": np.max(np.abs(own["components"] @ own["components"].T - np.eye(n_components))),
        "orthogonality_ref": np.max(np.abs(reference.components_ @ reference.components_.T - np.eye(n_components))),
    }
    metrics["equivalent"] = all((
        np.allclose(own["singular_values"], reference.singular_values_, atol=1e-8, rtol=1e-8),
        np.allclose(own["explained_variance_ratio"], reference.explained_variance_ratio_, atol=1e-8, rtol=1e-8),
        np.allclose(aligned_components, reference.components_, atol=1e-8, rtol=1e-8),
        np.allclose(aligned_scores, Z_ref, atol=1e-8, rtol=1e-8),
        np.allclose(own_reconstructed, ref_reconstructed, atol=1e-8, rtol=1e-8),
    ))

    print(f"\n3. СРАВНЕНИЕ С sklearn.PCA ({n_components} компонент)")
    print(f"{'PC':<7}{'Своя PCA':>16}{'sklearn PCA':>16}{'|Разность|':>16}{'Корреляция':>15}")
    print("-" * 70)
    for i in range(n_components):
        own_ratio = own["explained_variance_ratio"][i]
        ref_ratio = reference.explained_variance_ratio_[i]
        corr = np.corrcoef(aligned_scores[:, i], Z_ref[:, i])[0, 1]
        print(f"PC{i + 1:<5}{own_ratio:>16.8f}{ref_ratio:>16.8f}"
              f"{abs(own_ratio - ref_ratio):>16.2e}{corr:>15.8f}")

    print("\nПроверка численного совпадения (с учётом знака компонент):")
    captions = {
        "variance_max_diff": "Макс. разность собственных значений",
        "ratio_max_diff": "Макс. разность долей дисперсии",
        "singular_max_diff": "Макс. разность сингулярных значений",
        "components_max_diff": "Макс. разность направлений компонент",
        "scores_mae": "MAE преобразованных данных",
        "scores_rmse": "RMSE преобразованных данных",
        "scores_max_diff": "Макс. разность преобразованных данных",
        "mse_own": "MSE восстановления своей PCA",
        "mse_ref": "MSE восстановления sklearn PCA",
        "reconstruction_max_diff": "Макс. разность восстановленных данных",
        "orthogonality_own": "Ошибка ортогональности своей PCA",
        "orthogonality_ref": "Ошибка ортогональности sklearn PCA",
    }
    for key, caption in captions.items():
        print(f"{caption:<45}: {metrics[key]:.8e}")
    print(f"Эквивалентность реализаций: {metrics['equivalent']}")
    return aligned_scores, Z_ref, metrics


def plot_comparison(Z_own, Z_ref):
    n_show = min(3, Z_own.shape[1])
    fig, axes = plt.subplots(2, n_show, figsize=(5.2 * n_show, 9), squeeze=False)
    for i in range(n_show):
        ax = axes[0, i]
        ax.scatter(Z_own[:, i], Z_ref[:, i], s=12, alpha=0.45, label="Объекты")
        low = min(Z_own[:, i].min(), Z_ref[:, i].min())
        high = max(Z_own[:, i].max(), Z_ref[:, i].max())
        correlation = np.corrcoef(Z_own[:, i], Z_ref[:, i])[0, 1]
        ax.plot([low, high], [low, high], "--", label="Полное совпадение: y = x")
        ax.set(title=f"PC{i + 1}: сравнение проекций (r = {correlation:.6f})",
               xlabel="Собственная PCA", ylabel="sklearn PCA")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1, i]
        difference = Z_own[:, i] - Z_ref[:, i]
        ax.hist(difference, bins=40, label="Разность проекций")
        ax.axvline(0, linestyle="--", label="Нулевая разность")
        ax.axvline(difference.mean(), linestyle=":", label="Средняя разность")
        ax.set(title=f"PC{i + 1}: распределение разностей",
               xlabel="Собственная PCA − sklearn PCA", ylabel="Частота")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def pca_regression_analysis(X_raw, y, test_size=0.30, random_state=42):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    scaler = standardize_fit(X_train_raw)
    X_train = standardize_transform(X_train_raw, scaler)
    X_test = standardize_transform(X_test_raw, scaler)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_train_r2 = baseline.score(X_train, y_train)
    baseline_test_r2 = baseline.score(X_test, y_test)
    baseline_test_mse = np.mean((y_test - baseline.predict(X_test)) ** 2)

    print("\n4. ЛИНЕЙНАЯ РЕГРЕССИЯ БЕЗ PCA")
    print(f"Обучающая выборка: {len(X_train)}; тестовая выборка: {len(X_test)}")
    print(f"Train R²: {baseline_train_r2:.6f}")
    print(f"Test  R²: {baseline_test_r2:.6f}")
    print(f"Test MSE: {baseline_test_mse:.6f}")

    print("\n5. ЛИНЕЙНАЯ РЕГРЕССИЯ С СОБСТВЕННОЙ PCA")
    print(f"{'PC':>3} {'Train R²':>12} {'Test R²':>12} {'Train−Test':>12}"
          f" {'Доля дисперсии':>16} {'Test MSE':>12}")
    print("-" * 75)
    results = []

    for m in range(1, X_train.shape[1] + 1):
        pca_model, X_train_pca = fit_transform_pca(X_train, n_components=m)
        X_test_pca = transform_pca(X_test, pca_model)

        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        train_r2 = model.score(X_train_pca, y_train)
        test_r2 = model.score(X_test_pca, y_test)
        test_mse = np.mean((y_test - model.predict(X_test_pca)) ** 2)
        variance = pca_model["explained_variance_ratio"].sum()

        results.append({
            "n_components": m,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "r2_gap": train_r2 - test_r2,
            "explained_variance": variance,
            "test_mse": test_mse,
        })
        print(f"{m:>3} {train_r2:>12.6f} {test_r2:>12.6f}"
              f" {train_r2 - test_r2:>12.6f} {variance:>16.6f} {test_mse:>12.6f}")

    best = max(results, key=lambda row: row["test_r2"])
    print("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"Максимальный Test R² при {best['n_components']} компонентах")
    print(f"С PCA:  Test R²={best['test_r2']:.6f}, Test MSE={best['test_mse']:.6f}, "
          f"доля дисперсии={best['explained_variance']:.4%}")
    print(f"Без PCA: Test R²={baseline_test_r2:.6f}, Test MSE={baseline_test_mse:.6f}")
    print(f"Разница Test R² (с PCA − без PCA): {best['test_r2'] - baseline_test_r2:+.6f}")

    baseline_metrics = {
        "train_r2": baseline_train_r2,
        "test_r2": baseline_test_r2,
        "test_mse": baseline_test_mse,
    }
    return results, baseline_metrics, best


def plot_pca_regression(results, baseline, best):
    m = [row["n_components"] for row in results]
    train_r2 = [row["train_r2"] for row in results]
    test_r2 = [row["test_r2"] for row in results]
    test_mse = [row["test_mse"] for row in results]
    explained = [row["explained_variance"] for row in results]
    r2_gap = [row["r2_gap"] for row in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))

    ax1.plot(m, train_r2, "o-", label="Train R² (с PCA)")
    ax1.plot(m, test_r2, "s-", label="Test R² (с PCA)")
    ax1.axhline(baseline["test_r2"], linestyle="--", label="Test R² (без PCA)")
    ax1.axvline(best["n_components"], linestyle=":", label="Макс. Test R²")
    ax1.set(title="Качество линейной регрессии", xlabel="Количество компонент PCA", ylabel="R²")
    ax1.set_xticks(m)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(m, test_mse, "o-", label="Test MSE (с PCA)")
    ax2.axhline(baseline["test_mse"], linestyle="--", label="Test MSE (без PCA)")
    ax2.axvline(best["n_components"], linestyle=":", label="Макс. Test R²")
    ax2.set(title="Ошибка линейной регрессии", xlabel="Количество компонент PCA", ylabel="Test MSE")
    ax2.set_xticks(m)
    ax2.grid(alpha=0.3)
    ax2.legend()

    line1, = ax3.plot(m, explained, "o-", label="Накопленная доля дисперсии")
    threshold90 = ax3.axhline(0.90, linestyle="--", alpha=0.6, label="Порог 90%")
    threshold95 = ax3.axhline(0.95, linestyle=":", alpha=0.6, label="Порог 95%")
    ax3.set(title="Объяснённая дисперсия и разница Train−Test", xlabel="Количество компонент PCA",
            ylabel="Накопленная доля дисперсии")
    ax3.set_xticks(m)
    ax3.set_ylim(0, 1.05)
    ax3.grid(alpha=0.3)
    ax3_right = ax3.twinx()
    line2, = ax3_right.plot(m, r2_gap, "s-", label="Разница Train R² − Test R²")
    ax3_right.set_ylabel("Разница Train R² − Test R²")
    handles = [line1, threshold90, threshold95, line2]
    ax3.legend(handles[:3], [handle.get_label() for handle in handles[:3]], loc="upper left")
    ax3_right.legend([line2], [line2.get_label()], loc="lower right")

    fig.tight_layout()
    return fig


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X_raw, y, feature_names = load_data()
    print(f"Wine Quality Red: {X_raw.shape[0]} объектов, {X_raw.shape[1]} признаков")
    print(f"Признаки: {', '.join(feature_names)}")
    print("quality используется как целевая переменная только в линейной регрессии")

    scaler = standardize_fit(X_raw)
    X_scaled = standardize_transform(X_raw, scaler)

    full_pca = fit_pca_svd(X_scaled)
    dimensions, cumulative, _ = determine_effective_dimension(full_pca)
    print_pca_results(full_pca, dimensions, cumulative)
    fig1 = plot_explained_variance(full_pca, dimensions, cumulative)
    fig1.savefig(f"{PLOTS_DIR}/1_plot_explained_variance.png", dpi=300, bbox_inches="tight")

    m = dimensions[0.95]["dimension"]
    Z_own, Z_ref, metrics = compare_with_sklearn(X_scaled, n_components=m)
    fig2 = plot_comparison(Z_own, Z_ref)
    fig2.savefig(f"{PLOTS_DIR}/2_plot_comparison_pca.png", dpi=300, bbox_inches="tight")

    regression_results, baseline, best = pca_regression_analysis(X_raw, y)
    fig3 = plot_pca_regression(regression_results, baseline, best)
    fig3.savefig(f"{PLOTS_DIR}/3_plot_pca_regression.png", dpi=300, bbox_inches="tight")

    print(f"\nГрафики сохранены в: {PLOTS_DIR}")
    if not metrics["equivalent"]:
        raise AssertionError("Численная проверка эквивалентности со sklearn не пройдена")
    plt.show()


if __name__ == "__main__":
    main()
