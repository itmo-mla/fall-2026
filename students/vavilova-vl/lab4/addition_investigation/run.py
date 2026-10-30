"""Дополнительное исследование: PCA, PCR и гребневая регрессия на данных испытаний газоконденсатных скважин.

Использует реализацию PCA и проверки из лабораторной (lab4/source).
Запуск из каталога lab4:  uv run python addition_investigation/run.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

import gas_plots
import plots
from dimension import condition_number, effective_dimension, residual_variance, steep_slope_dimension
from gas_data import Preprocessor, load_gas_data
from models import (SVDFit, bootstrap_coefficients, cv_rmse, effective_dimension_ridge,
                    grouped_folds, random_folds)
from pca import PCA
from reference import check_theorem, compare, reconstruction_errors

IMAGES = HERE / "images"
EPS = 0.05
TOLERANCE = 1e-10
N_SPLITS, N_REPEATS = 5, 10
TAUS = np.logspace(-3, 4, 71)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    data = load_gas_data()
    X, y, groups, names = data.X, data.y, data.groups, data.feature_names
    n_objects, n_features = data.shape
    Xs = Preprocessor().fit_transform(X, groups)

    section("Датасет")
    print(f"Строк в файле: {data.n_raw}; после очистки ℓ = {n_objects}, n = {n_features}, "
          f"серий испытаний: {len(np.unique(groups))}")
    for fix in data.fixes:
        print(" -", fix)
    print(f"y = дебит газа: среднее {y.mean():.1f}, std {y.std():.1f}, от {y.min():.1f} до {y.max():.1f}")
    C = np.corrcoef(Xs, rowvar=False)
    pairs = sorted(((abs(C[i, j]), names[i], names[j], C[i, j])
                    for i in range(n_features) for j in range(i + 1, n_features)), reverse=True)
    print("Сильнее всего коррелированные пары:")
    for _, a, b, r in pairs[:8]:
        print(f"  {a:>9} – {b:<9} r = {r:+.3f}")
    r_y = [np.corrcoef(Xs[:, j], y)[0, 1] for j in range(n_features)]
    print("Корреляция с y:", ", ".join(f"{n} {r:+.2f}" for n, r in zip(names, r_y)))

    section("PCA через SVD")
    pca = PCA().fit(Xs)
    lam = pca.all_eigenvalues_
    E = residual_variance(lam)
    print(f"{'j':>2} {'λ_j':>9} {'доля':>7} {'накоп.':>7} {'E_j':>8}")
    for j in range(n_features):
        print(f"{j + 1:>2} {lam[j]:9.3f} {pca.explained_variance_ratio_[j]:7.2%} {1 - E[j + 1]:7.2%} {E[j + 1]:8.5f}")
    print("Главные направления (наибольшие по модулю координаты):")
    for j in range(n_features):
        u = pca.components_[j]
        top = np.argsort(-np.abs(u))[:4]
        print(f"  PC{j + 1:<2}", ", ".join(f"{names[k]} {u[k]:+.2f}" for k in top))

    section("Основная теорема PCA и сравнение с sklearn")
    worst_theorem = max(check_theorem(Xs, m).max_error for m in range(1, n_features + 1))
    comparisons = [compare(Xs, m) for m in range(1, n_features + 1)]
    worst_sklearn = max(c.max_error for c in comparisons)
    flips = sum(c.sign_flips for c in comparisons)
    print(f"Свойства 1–4, max отклонение: {worst_theorem:.2e}  [{'OK' if worst_theorem < TOLERANCE else 'НАРУШЕНО'}]")
    print(f"sklearn PCA, max отклонение:  {worst_sklearn:.2e}  [{'OK' if worst_sklearn < TOLERANCE else 'РАСХОЖДЕНИЕ'}], "
          f"смен знака: {flips}")
    rec_ours, rec_ref = reconstruction_errors(Xs)
    print(f"max |ошибка восстановления − E_m| = {np.max(np.abs(rec_ours - E[1:])):.2e}")

    section("Эффективная размерность и обусловленность")
    m_eff = effective_dimension(lam, EPS)
    for eps in (0.2, 0.1, EPS, 0.01):
        print(f"E_m ≤ {eps}: m = {effective_dimension(lam, eps)}")
    ratios = E[:-2] / E[1:-1]
    print("E_{m−1} / E_m:", "  ".join(f"{m}: {r:.2f}" for m, r in enumerate(ratios, start=1)))
    print(f"«Крутой склон»: m = {steep_slope_dimension(lam)}")
    mu = condition_number(lam)
    print(f"μ(FᵀF) = λ_max / λ_min = {mu:.0f}")
    for m in range(n_features, 0, -1):
        if lam[0] / lam[m - 1] < 100:
            print(f"μ < 100 впервые при m ≤ {m}: μ = {lam[0] / lam[m - 1]:.1f}")
            break

    section("Регрессия: МНК, PCR, гребневая (10×5-fold CV)")
    pcr_solvers = [lambda s, m=m: s.pcr(m) for m in range(1, n_features + 1)]
    ridge_solvers = [lambda s, t=t: s.ridge(t) for t in TAUS]
    const = [lambda s: (np.zeros(n_features), s.y_mean)]

    folds = lambda: grouped_folds(groups, N_SPLITS, N_REPEATS)
    rmse_pcr = cv_rmse(X, y, groups, folds(), pcr_solvers)
    rmse_ridge = cv_rmse(X, y, groups, folds(), ridge_solvers)
    rmse_const = cv_rmse(X, y, groups, folds(), const)[0]
    rmse_pcr_random = cv_rmse(X, y, groups, random_folds(n_objects, N_SPLITS, N_REPEATS), pcr_solvers)

    print(f"Константа ȳ: RMSE = {rmse_const:.2f}")
    print(f"{'m':>3} {'по сериям':>10} {'по замерам':>11}")
    for m in range(1, n_features + 1):
        mark = "  ← m_eff" if m == m_eff else ""
        print(f"{m:>3} {rmse_pcr[m - 1]:10.2f} {rmse_pcr_random[m - 1]:11.2f}{mark}")
    m_best = int(np.argmin(rmse_pcr)) + 1
    k_best = int(np.argmin(rmse_ridge))
    tau_best = TAUS[k_best]
    n_eff_curve = np.array([effective_dimension_ridge(lam, t) for t in TAUS])
    print(f"МНК (m = n): {rmse_pcr[-1]:.2f}; лучший PCR: m = {m_best}, RMSE = {rmse_pcr[m_best - 1]:.2f}")
    print(f"Лучшая гребневая: τ = {tau_best:.4g}, RMSE = {rmse_ridge[k_best]:.2f}, "
          f"n_eff = {n_eff_curve[k_best]:.2f}")
    print(f"μ(FᵀF + τ*I) = {(lam[0] + tau_best) / (lam[-1] + tau_best):.1f}")

    section("Проверка МНК и гребневой регрессии против sklearn")
    svd = SVDFit.fit(Xs, y)
    alpha_ols, b_ols = svd.ols()
    ref = LinearRegression().fit(Xs, y)
    print(f"МНК:       max |α − α_sk| = {np.max(np.abs(alpha_ols - ref.coef_)):.2e}, "
          f"|w0 − w0_sk| = {abs(b_ols - ref.intercept_):.2e}")
    alpha_ridge, b_ridge = svd.ridge(tau_best)
    ref = Ridge(alpha=tau_best).fit(Xs, y)
    print(f"Гребневая: max |α − α_sk| = {np.max(np.abs(alpha_ridge - ref.coef_)):.2e}, "
          f"|w0 − w0_sk| = {abs(b_ridge - ref.intercept_):.2e}")
    alpha_pcr, _ = svd.pcr(m_best)

    section("Коэффициенты на всей выборке (признаки стандартизованы)")
    print(f"{'признак':>9} {'МНК':>9} {'PCR':>9} {'Ridge':>9}")
    for k, name in enumerate(names):
        print(f"{name:>9} {alpha_ols[k]:9.1f} {alpha_pcr[k]:9.1f} {alpha_ridge[k]:9.1f}")
    print(f"{'‖α‖':>9} {np.linalg.norm(alpha_ols):9.1f} {np.linalg.norm(alpha_pcr):9.1f} "
          f"{np.linalg.norm(alpha_ridge):9.1f}")

    section("Бутстреп серий испытаний: разброс коэффициентов")
    solvers = [lambda s: s.ols(), lambda s: s.pcr(m_best), lambda s: s.ridge(tau_best)]
    labels = ["МНК", f"PCR, m = {m_best}", f"Ridge, τ = {tau_best:.3g}"]
    boot = bootstrap_coefficients(X, y, groups, solvers)
    spread = np.percentile(boot, 95, axis=0) - np.percentile(boot, 5, axis=0)
    print(f"{'признак':>9}" + "".join(f"{l:>18}" for l in labels))
    for k, name in enumerate(names):
        print(f"{name:>9}" + "".join(f"{spread[i, k]:18.1f}" for i in range(len(labels))))
    print(f"{'медиана':>9}" + "".join(f"{np.median(spread[i]):18.1f}" for i in range(len(labels))))
    sign_flip = np.mean(np.sign(boot) != np.sign(np.median(boot, axis=0)), axis=0)
    print("Доля бутстреп-выборок, где знак коэффициента противоположен медианному (в среднем по признакам):",
          ", ".join(f"{l}: {sign_flip[i].mean():.1%}" for i, l in enumerate(labels)))

    plots.spectrum(lam, E, EPS, m_eff, IMAGES / "spectrum.png")
    plots.loadings(pca.components_, names, IMAGES / "loadings.png")
    plots.reconstruction(rec_ours, rec_ref, E, IMAGES / "reconstruction.png")
    gas_plots.correlation(Xs, names, IMAGES / "correlation.png")
    gas_plots.projection(pca.transform(Xs), y, groups, pca.explained_variance_ratio_, IMAGES / "projection.png")
    gas_plots.pcr_curves(rmse_pcr, rmse_pcr_random, rmse_const, m_eff, IMAGES / "pcr.png")
    gas_plots.ridge_curve(TAUS, rmse_ridge, n_eff_curve, rmse_pcr[-1], rmse_pcr[m_best - 1], IMAGES / "ridge.png")
    gas_plots.coefficients(boot, labels, names, IMAGES / "coefficients.png")
    print(f"\nГрафики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
