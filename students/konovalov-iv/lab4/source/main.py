from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import r2_score

from dataset import load_data
from pca import PCA
from plots import plot_components, plot_reconstruction, plot_regression, plot_spectrum

PLOTS = Path(__file__).resolve().parent / "plots"


def spectral_ols(pca, y, m):
    d, U, V = pca.singular_values_[:m], pca.components_[:m].T, pca.V_[:, :m]
    return U @ ((V.T @ (y - y.mean())) / d)


def main():
    PLOTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, names = load_data()
    n = X_train.shape[1]
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"число обусловленности: {np.linalg.cond(X_train):.2f}")

    pca = PCA().fit(X_train)
    ratio = pca.explained_variance_ratio_
    print("доли дисперсии:", " ".join(f"{r:.3f}" for r in ratio))

    U, G = pca.components_.T, pca.transform(X_train)
    print(f"\nUᵀU = I: {np.abs(U.T @ U - np.eye(n)).max():.1e}")
    print(f"GᵀG = Λ: {np.abs(G.T @ G - np.diag(pca.eigenvalues_)).max():.1e}")
    errors, tails = [], []
    centered = X_train - pca.mean_
    for m in range(n + 1):
        approx = pca.transform(X_train, m) @ pca.components_[:m]
        errors.append(np.sum((approx - centered) ** 2))
        tails.append(pca.eigenvalues_[m:].sum())
    print(f"||GUᵀ - F||² = Σ отброшенных λ: {np.allclose(errors, tails)}")

    residual = pca.residual_share()
    print("\nm   E_m     E_(m-1)/E_m")
    slope = pca.steep_slope()
    for m in range(1, n + 1):
        print(f"{m}   {residual[m]:.4f}  {slope[m - 1]:.2f}" if m < n else f"{m}   {residual[m]:.4f}")
    for eps in (0.15, 0.05, 0.01):
        print(f"эффективная размерность при eps = {eps}: {pca.effective_dimension(eps)}")
    print(f"критерий Кайзера (λ > среднего): {int(np.sum(pca.explained_variance_ > 1))}")

    ref = SkPCA(svd_solver="full").fit(X_train)
    sign = np.sign(np.sum(ref.components_ * pca.components_, axis=1))
    print("\nсравнение с sklearn.decomposition.PCA, max |разница|:")
    print(f"  mean_                     {np.abs(pca.mean_ - ref.mean_).max():.1e}")
    print(f"  explained_variance_       {np.abs(pca.explained_variance_ - ref.explained_variance_).max():.1e}")
    print(f"  components_ (знак учтён)  {np.abs(pca.components_ * sign[:, None] - ref.components_).max():.1e}")
    print(f"  знаки осей совпали у {int(np.sum(sign > 0))} из {n}")
    print(f"  transform (знак учтён)    {np.abs(pca.transform(X_test) * sign - ref.transform(X_test)).max():.1e}")

    ms = np.arange(1, n + 1)
    r2_train, r2_test = [], []
    for m in ms:
        alpha = spectral_ols(pca, y_train, m)
        r2_train.append(r2_score(y_train, (X_train - pca.mean_) @ alpha + y_train.mean()))
        r2_test.append(r2_score(y_test, (X_test - pca.mean_) @ alpha + y_train.mean()))
    alpha_ols = np.linalg.lstsq(X_train, y_train - y_train.mean(), rcond=None)[0]
    r2_ols = r2_score(y_test, X_test @ alpha_ols + y_train.mean())
    print(f"\nR² на тесте, МНК по всем признакам: {r2_ols:.4f}")
    print("m  R² train  R² test")
    for m, a, b in zip(ms, r2_train, r2_test):
        print(f"{m}  {a:.4f}    {b:.4f}")
    print(f"МНК из SVD при m = n совпадает с lstsq: "
          f"{np.allclose(spectral_ols(pca, y_train, n), alpha_ols)}")

    plot_spectrum(pca.eigenvalues_, ratio, residual, PLOTS / "spectrum.png")
    plot_components(pca.components_, names, ratio, PLOTS / "components.png")
    plot_reconstruction(errors, tails, PLOTS / "reconstruction.png")
    plot_regression(ms, r2_train, r2_test, r2_ols, PLOTS / "regression.png")


if __name__ == "__main__":
    main()
