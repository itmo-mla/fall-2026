import json
from pathlib import Path

import numpy as np

from dataset import load_dataset
from effective_dim import (effective_rank, n_components_for_variance,
                           n_components_kaiser)
from pca import PCASVD
from plots import plot_cumulative_variance, plot_scree


def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def compare_with_reference(pca_own: PCASVD, X: np.ndarray) -> dict:
    from sklearn.decomposition import PCA

    ref = PCA(svd_solver="full", n_components=pca_own.components_.shape[0])
    ref.fit(X)

    m_own = np.abs(pca_own.components_)
    m_ref = np.abs(ref.components_)
    max_comp_diff = float(np.max(np.abs(m_own - m_ref)))

    ev_diff = float(np.max(np.abs(pca_own.explained_variance_
                                  - ref.explained_variance_)))
    ratio_diff = float(np.max(np.abs(pca_own.explained_variance_ratio_
                                     - ref.explained_variance_ratio_)))

    proj_own = np.abs(pca_own.transform(X))
    proj_ref = np.abs(ref.transform(X))
    max_proj_diff = float(np.max(np.abs(proj_own - proj_ref)))

    k_rec = 20
    ref_rec = PCA(svd_solver="full", n_components=k_rec).fit(X)
    rec_own = pca_own.inverse_transform(pca_own.transform(X)[:, :k_rec])
    rec_ref = ref_rec.inverse_transform(ref_rec.transform(X))
    max_rec_diff = float(np.max(np.abs(rec_own - rec_ref)))

    return {
        "n_features": int(X.shape[1]),
        "max_components_diff": max_comp_diff,
        "max_explained_variance_diff": ev_diff,
        "max_ratio_diff": ratio_diff,
        "max_projection_diff": max_proj_diff,
        "max_reconstruction_diff": max_rec_diff,
    }


def main() -> None:
    np.random.seed(42)

    print("=== Загрузка датасета ===")
    df_raw, X_raw, feature_names, y = load_dataset()
    print(f"Строк: {X_raw.shape[0]}, признаков после one-hot: {X_raw.shape[1]}")

    X, mean, std = standardize(X_raw)
    print(f"[OK] Стандартизация выполнена (ddof=0).")

    print("\n=== Собственная PCA через SVD ===")
    pca = PCASVD()
    Z = pca.fit_transform(X)
    print(f"[OK] Спектр: {len(pca.singular_values_)} компонент; "
          f"S[0] = {pca.singular_values_[0]:.2f}")

    print("\n=== Эффективная размерность ===")
    k_95 = n_components_for_variance(pca.explained_variance_ratio_, 0.95)
    k_90 = n_components_for_variance(pca.explained_variance_ratio_, 0.90)
    k_99 = n_components_for_variance(pca.explained_variance_ratio_, 0.99)
    kaiser = n_components_kaiser(pca.explained_variance_)
    rank_eff = effective_rank(pca.singular_values_)
    print(f"  90% дисперсии : {k_90} компонент")
    print(f"  95% дисперсии : {k_95} компонент")
    print(f"  99% дисперсии : {k_99} компонент")
    print(f"  Кайзер (ev>1) : {kaiser} компонент")
    print(f"  Эффективный ранг (энтропия): {rank_eff:.2f}")

    print("\n=== Проверка эквивалентности с эталонной реализацией ===")
    report = compare_with_reference(pca, X)
    for key, value in report.items():
        print(f"  {key}: {value:.3e}")

    print("\n=== Графики ===")
    path_scree = plot_scree(pca.explained_variance_, kaiser, k_95)
    path_cum = plot_cumulative_variance(pca.explained_variance_ratio_, k_95,
                                        rank_eff)
    print(f"  {path_scree}")
    print(f"  {path_cum}")

    summary = {
        "dataset": "avikasliwal/used-cars-price-prediction (train-data.csv)",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "k_90": k_90, "k_95": k_95, "k_99": k_99,
        "kaiser": kaiser,
        "effective_rank": rank_eff,
        "equivalence": report,
    }
    summary_path = Path(__file__).resolve().parent / "pics" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== Готово, сводка: {summary_path} ===")


if __name__ == "__main__":
    main()
