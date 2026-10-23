from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

from data import load_data, prepare
from pca import PCA, effective_dimension


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    x_raw, labels, class_names = load_data()
    rows, dimensions = [], []
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    projection_figure, projection_axes = plt.subplots(1, 2, figsize=(12, 5))

    for row, standardize in enumerate((False, True)):
        mode = "Стандартизация" if standardize else "Центрирование"
        x, _, mean, scale = prepare(x_raw, x_raw, standardize)
        full = PCA(4).fit(x)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        for threshold in (0.90, 0.95, 0.99):
            dimensions.append({
                "preprocessing": mode,
                "threshold": threshold,
                "dimension": effective_dimension(full.explained_variance_ratio_, threshold),
            })

        for k in range(1, 5):
            model = PCA(k).fit(x)
            reference = SklearnPCA(n_components=k, svd_solver="full").fit(x)
            z = model.transform(x)
            z_ref = reference.transform(x)
            restored = model.inverse_transform(z)
            restored_ref = reference.inverse_transform(z_ref)
            signs = np.where(np.sum(model.components_ * reference.components_, axis=1) < 0, -1, 1)
            projector = model.components_.T @ model.components_
            projector_ref = reference.components_.T @ reference.components_
            variance_error = np.max(np.abs(model.explained_variance_ - reference.explained_variance_))
            ratio_error = np.max(np.abs(model.explained_variance_ratio_ - reference.explained_variance_ratio_))
            projection_error = np.max(np.abs(projector - projector_ref))
            score_error = np.max(np.abs(z * signs - z_ref))
            reconstruction_error = np.max(np.abs(restored - restored_ref))
            np.testing.assert_allclose(model.singular_values_, reference.singular_values_, atol=1e-10)
            np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, atol=1e-10)
            np.testing.assert_allclose(model.explained_variance_ratio_, reference.explained_variance_ratio_, atol=1e-10)
            np.testing.assert_allclose(projector, projector_ref, atol=1e-10)
            np.testing.assert_allclose(z * signs, z_ref, atol=1e-10)
            np.testing.assert_allclose(restored, restored_ref, atol=1e-10)
            rows.append({
                "preprocessing": mode,
                "n_components": k,
                "retained_variance": cumulative[k - 1],
                "mse": np.mean((x - restored)**2),
                "mse_cm2": np.mean((x_raw - (restored * scale + mean))**2),
                "reference_mse": np.mean((x - restored_ref)**2),
                "reference_mse_cm2": np.mean((x_raw - (restored_ref * scale + mean))**2),
                "variance_max_error": variance_error,
                "ratio_max_error": ratio_error,
                "projector_max_error": projection_error,
                "scores_max_error": score_error,
                "reconstruction_max_error": reconstruction_error,
            })

        subset = pd.DataFrame(rows).tail(4)
        components = np.arange(1, 5)
        axes[row, 0].bar(components, full.explained_variance_ratio_, label="Доля компоненты")
        axes[row, 0].plot(components, cumulative, "o-", color="tab:orange", label="Накопленная доля")
        axes[row, 0].axhline(0.95, color="red", linestyle="--", label="Порог 95%")
        axes[row, 0].set_ylabel("Доля дисперсии")
        axes[row, 1].plot(components, subset["mse_cm2"], "o-", label="Собственная PCA")
        axes[row, 1].plot(components, subset["reference_mse_cm2"], "x", markersize=9, label="Sklearn PCA")
        axes[row, 1].set_ylabel("MSE восстановления, см²")
        axes[row, 2].plot(components, full.explained_variance_, "o-", label="Собственные значения")
        axes[row, 2].set_ylabel("Дисперсия компоненты")
        for ax in axes[row]:
            ax.set_title(mode)
            ax.set_xlabel("Число компонент / номер компоненты")
            ax.set_xticks(components)
            ax.grid(alpha=0.25)
            ax.legend()

        z = PCA(2).fit_transform(x)
        for index, name in enumerate(class_names):
            mask = labels == index
            projection_axes[row].scatter(z[mask, 0], z[mask, 1], label=name, alpha=0.75)
        projection_axes[row].set(title=mode, xlabel="PC1", ylabel="PC2")
        projection_axes[row].grid(alpha=0.25)
        projection_axes[row].legend()

    figure.tight_layout()
    figure.savefig(root / "figures" / "pca_comparison.png", dpi=160)
    projection_figure.tight_layout()
    projection_figure.savefig(root / "figures" / "projections.png", dpi=160)
    plt.close("all")
    results = pd.DataFrame(rows)
    dimension_table = pd.DataFrame(dimensions)
    results.to_csv(root / "results" / "pca.tsv", sep="\t", index=False)
    dimension_table.to_csv(root / "results" / "dimensions.tsv", sep="\t", index=False)
    print(results.to_string(index=False))
    print(dimension_table.to_string(index=False))


if __name__ == "__main__":
    main()
