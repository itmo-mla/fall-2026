"""Лабораторная работа №4: PCA через SVD.

Запуск из каталога lab4:  uv run python source/main.py
"""

from pathlib import Path

import numpy as np

import plots
from data import load_dataset
from dimension import condition_number, effective_dimension, residual_variance, steep_slope_dimension
from pca import PCA
from reference import check_theorem, compare, reconstruction_errors
from regression import pcr_curve, pcr_fit

IMAGES = Path(__file__).resolve().parent.parent / "images"
EPS = 0.05
TOLERANCE = 1e-10


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    data = load_dataset()
    X, y = data.X, data.y
    n_objects, n_features = data.shape

    section("Датасет")
    print(f"Titanic: ℓ = {n_objects}, n = {n_features}")
    print("Признаки:", ", ".join(data.feature_names))

    section("PCA через SVD: F = V D Uᵀ")
    pca = PCA().fit(X)
    lam = pca.all_eigenvalues_
    E = residual_variance(lam)
    print(f"{'j':>2} {'√λ_j':>8} {'λ_j':>9} {'λ_j/(ℓ−1)':>10} {'доля':>7} {'накоп.':>7} {'E_j':>7}")
    for j in range(n_features):
        print(f"{j + 1:>2} {pca.singular_values_[j]:8.3f} {lam[j]:9.2f} {pca.explained_variance_[j]:10.4f} "
              f"{pca.explained_variance_ratio_[j]:7.2%} {1 - E[j + 1]:7.2%} {E[j + 1]:7.4f}")
    # λ_j из SVD должны совпадать с собственными значениями FᵀF.
    F = X - X.mean(axis=0)
    gram_eig = np.sort(np.linalg.eigvalsh(F.T @ F))[::-1]
    print(f"max |λ_SVD − λ_eig(FᵀF)| / λ_1 = {np.max(np.abs(lam - gram_eig)) / lam[0]:.2e}")

    section("Основная теорема PCA (свойства 1–4)")
    print(f"{'m':>2} {'UᵀU−I':>9} {'GᵀG−Λ':>9} {'FᵀFU−UΛ':>9} {'‖GUᵀ−F‖²−Σλ':>13}")
    worst = 0.0
    for m in range(1, n_features + 1):
        t = check_theorem(X, m)
        worst = max(worst, t.max_error)
        print(f"{m:>2} {t.orthonormal_u:9.1e} {t.orthogonal_g:9.1e} {t.eigen_u:9.1e} {t.residual:13.1e}")
    print(f"Максимальное отклонение: {worst:.2e}  [{'OK' if worst < TOLERANCE else 'НАРУШЕНО'}]")

    section("Эффективная размерность")
    m_eff = effective_dimension(lam, EPS)
    print(f"Критерий E_m ≤ ε = {EPS}:  m = {m_eff}")
    for eps in (0.2, 0.1, 0.01):
        print(f"  при ε = {eps}: m = {effective_dimension(lam, eps)}")
    ratios = E[:-2] / E[1:-1]
    print("E_{m−1} / E_m:", "  ".join(f"m={m}: {r:.2f}" for m, r in enumerate(ratios, start=1)))
    print(f"Критерий «крутого склона» (max E_(m−1)/E_m): m = {steep_slope_dimension(lam)}")
    print(f"Число обусловленности μ(FᵀF) = λ_max / λ_min = {condition_number(lam):.1f}")

    section("Сравнение с sklearn.decomposition.PCA")
    print(f"{'m':>2} {'U':>9} {'λ/(ℓ−1)':>9} {'доля':>9} {'√λ':>9} {'G':>9} {'F̂':>9} {'знаки':>6}")
    worst = 0.0
    for m in range(1, n_features + 1):
        c = compare(X, m)
        worst = max(worst, c.max_error)
        print(f"{m:>2} {c.components:9.1e} {c.explained_variance:9.1e} {c.explained_variance_ratio:9.1e} "
              f"{c.singular_values:9.1e} {c.transformed:9.1e} {c.reconstructed:9.1e} {c.sign_flips:>6}")
    status = "OK" if worst < TOLERANCE else "РАСХОЖДЕНИЕ"
    print(f"Максимальное отклонение по всем m: {worst:.2e}  [{status}, допуск {TOLERANCE:g}]")
    rec_ours, rec_ref = reconstruction_errors(X)
    print(f"max |ошибка восстановления ours − sklearn| = {np.max(np.abs(rec_ours - rec_ref)):.2e}")
    print(f"max |ошибка восстановления − E_m|          = {np.max(np.abs(rec_ours - E[1:])):.2e}")

    section("Регрессия на главных компонентах (10×5-fold CV)")
    rmse = pcr_curve(X, y)
    for m, r in enumerate(rmse, start=1):
        mark = "  ← m_eff" if m == m_eff else ""
        print(f"m = {m:>2}: RMSE = {r:7.4f}{mark}")
    alpha_full, _ = pcr_fit(X, y, n_features)
    alpha_pcr, _ = pcr_fit(X, y, m_eff)
    print(f"‖α*‖ МНК на всех признаках = {np.linalg.norm(alpha_full):.3f}, "
          f"‖α*‖ по {m_eff} компонентам = {np.linalg.norm(alpha_pcr):.3f}")

    plots.spectrum(lam, E, EPS, m_eff, IMAGES / "spectrum.png")
    plots.projection(pca.transform(X), y, pca.explained_variance_ratio_, IMAGES / "projection.png")
    plots.loadings(pca.components_, data.feature_names, IMAGES / "loadings.png")
    plots.reconstruction(rec_ours, rec_ref, E, IMAGES / "reconstruction.png")
    plots.pcr(rmse, m_eff, IMAGES / "pcr.png")
    print(f"\nГрафики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
