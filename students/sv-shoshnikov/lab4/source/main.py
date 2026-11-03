from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pca import PCASVD, effective_dimension, kaiser, steep_slope
from plots import (plot_correlations, plot_loadings, plot_projection, plot_regression_k,
                   plot_ridge, plot_spectrum)
from regression import SVDRegression, cv_scores, gcv_scores

SEED = 42
OUT = Path(__file__).resolve().parent.parent


def load(name):
    d = fetch_california_housing() if name == "California Housing" else load_diabetes()
    X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.3, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    return dict(features=list(d.feature_names), X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                X_train_std=scaler.transform(X_tr), X_test_std=scaler.transform(X_te))


def align_signs(ours, ref):
    signs = np.sign(np.sum(ours * ref, axis=1))
    return ours * signs[:, None], int(np.sum(signs < 0))


def pca_section(d):
    X, Xt = d["X_train_std"], d["X_test_std"]
    pca = PCASVD().fit(X)
    ref = PCA(svd_solver="full").fit(X)
    comps, flips = align_signs(pca.components_, ref.components_)
    n_feat = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    pairs = sorted(((abs(corr[i, j]), i, j) for i in range(n_feat) for j in range(i + 1, n_feat)), reverse=True)[:3]
    print(f"   Признаков {n_feat}, число обусловленности cond(X) = {np.linalg.cond(X):.2f}")
    print("   Самые сильные корреляции: " + ", ".join(
        f"{d['features'][i]}-{d['features'][j]} ({corr[i, j]:+.3f})" for _, i, j in pairs))

    print("\n   Собственные свойства разложения:")
    Z = pca.transform(X)
    print(f"   ортонормированность осей |V V^T - I|max = {np.abs(pca.components_ @ pca.components_.T - np.eye(n_feat)).max():.1e}")
    off = np.corrcoef(Z, rowvar=False) - np.eye(n_feat)
    print(f"   некоррелированность проекций max|corr| = {np.abs(off).max():.1e}")
    print(f"   дисперсия проекций = s^2/(n-1): max|diff| = {np.abs(Z.var(axis=0, ddof=1) - pca.explained_variance_).max():.1e}")
    print(f"   восстановление по всем осям: max|diff| = {np.abs(pca.inverse_transform(Z) - X).max():.1e}")

    print("\n   Сравнение с sklearn.decomposition.PCA (svd_solver='full'):")
    print(f"   mean_:                       {np.abs(pca.mean_ - ref.mean_).max():.1e}")
    print(f"   singular_values_:            {np.abs(pca.singular_values_ - ref.singular_values_).max():.1e}")
    print(f"   explained_variance_:         {np.abs(pca.explained_variance_ - ref.explained_variance_).max():.1e}")
    print(f"   explained_variance_ratio_:   {np.abs(pca.explained_variance_ratio_ - ref.explained_variance_ratio_).max():.1e}")
    print(f"   components_ (знаки выровнены, различались у {flips} из {n_feat}): {np.abs(comps - ref.components_).max():.1e}")
    sign = np.where(np.sum(pca.components_ * ref.components_, axis=1) < 0, -1.0, 1.0)
    print(f"   transform на тесте (с учётом знака): {np.abs(pca.transform(Xt) * sign - ref.transform(Xt)).max():.1e}")
    for k in sorted({2, max(2, n_feat // 2), n_feat - 1}):
        ours = pca.inverse_transform(pca.transform(Xt, k))
        ref_k = PCA(n_components=k, svd_solver="full").fit(X)
        theirs = ref_k.inverse_transform(ref_k.transform(Xt))
        print(f"   inverse_transform при k = {k}: {np.abs(ours - theirs).max():.1e}")

    E = pca.residual_share()
    print("\n   Эффективная размерность:")
    print("   m      " + " ".join(f"{m:>6}" for m in range(1, n_feat + 1)))
    print("   доля   " + " ".join(f"{r:6.3f}" for r in pca.explained_variance_ratio_))
    print("   E_m    " + " ".join(f"{e:6.3f}" for e in E[1:]))
    for k in sorted({2, max(2, n_feat // 2), n_feat - 1}):
        direct = np.sum((X - pca.inverse_transform(pca.transform(X, k))) ** 2) / np.sum(X ** 2)
        print(f"   проверка: ||X - X_k||^2 / ||X||^2 при k = {k}: {direct:.6f}, E_k = {E[k]:.6f}")
    d.update(pca=pca, E=E, m_eff=effective_dimension(E, 0.05))
    print(f"   eps = 0.05: m = {d['m_eff']};  eps = 0.01: m = {effective_dimension(E, 0.01)};  "
          f"крутой склон (в 3 раза): m = {steep_slope(E)};  Кайзер: m = {kaiser(pca.explained_variance_)}")


def regression_section(d):
    Xtr, Xte, ytr, yte = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    n_feat = Xtr.shape[1]
    ks = np.arange(1, n_feat + 1)
    taus = np.logspace(-2, 4, 13)

    pcr_cv = cv_scores(Xtr, ytr, "pcr", ks)
    pcr_test = np.array([r2_score(yte, SVDRegression("pcr", k).fit(Xtr, ytr).predict(Xte)) for k in ks])
    ridge_cv = cv_scores(Xtr, ytr, "ridge", taus)
    ols_cv = cv_scores(Xtr, ytr, "ols", [None])[0]
    k_best, tau_best = int(ks[np.argmax(pcr_cv)]), float(taus[np.argmax(ridge_cv)])
    tau_gcv = float(taus[np.argmin(gcv_scores(Xtr, ytr, taus))])

    print("\nPCR: R2 в зависимости от числа компонент k")
    print("   k        " + " ".join(f"{k:>6}" for k in ks))
    print("   R2 CV    " + " ".join(f"{v:6.3f}" for v in pcr_cv))
    print("   R2 тест  " + " ".join(f"{v:6.3f}" for v in pcr_test))
    models = {"МНК (все признаки)": SVDRegression("ols"),
              f"Ridge (tau = {tau_best:.3g})": SVDRegression("ridge", tau_best),
              f"PCR (k = {k_best})": SVDRegression("pcr", k_best)}
    print("\n   Регрессия (подбор по 5-fold CV на train, тест используется один раз):")
    print(f"   {'модель':<22} {'R2 тест':>8} {'RMSE тест':>10} {'||w||':>8} {'след сглаживания':>17}")
    fitted = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        pred = m.predict(Xte)
        print(f"   {name:<22} {r2_score(yte, pred):>8.4f} {mean_squared_error(yte, pred) ** 0.5:>10.4f} "
              f"{np.linalg.norm(m.w_):>8.3f} {m.degrees_of_freedom():>17.3f}")
    d.update(ks=ks, pcr_cv=pcr_cv, pcr_test=pcr_test, ols_test=r2_score(yte, models["МНК (все признаки)"].predict(Xte)),
             k_best=k_best, tau_gcv=tau_gcv, taus=taus, ridge_cv=ridge_cv, tau_best=tau_best, ols_cv=ols_cv)
    print(f"   Выбор по CV: k* = {k_best} (R2 {pcr_cv.max():.4f}), tau* = {tau_best:.3g} (R2 {ridge_cv.max():.4f}), "
          f"МНК R2 {ols_cv:.4f}")
    print(f"   tau по GCV (без разбиения на блоки): {tau_gcv:.3g}")

    ols, ridge, pcr = fitted.values()
    Z = ols.transform(Xtr)
    yc = ytr - ytr.mean()
    w_ne = np.linalg.solve(Z.T @ Z, Z.T @ yc)
    print("\n   Эквивалентность с эталонами (по предсказаниям на тесте и коэффициентам):")
    print(f"   МНК: через SVD и нормальная система: max|dw| = {np.abs(ols.w_ - w_ne).max():.1e}; "
          f"cond(Z) = {np.linalg.cond(Z):.1f}, cond(Z^T Z) = {np.linalg.cond(Z.T @ Z):.1f}")
    ref_ols = LinearRegression().fit(Z, ytr)
    print(f"   МНК vs sklearn LinearRegression: max|dw| = {np.abs(ols.w_ - ref_ols.coef_).max():.1e}, "
          f"max|dpred| = {np.abs(ols.predict(Xte) - ref_ols.predict(ols.transform(Xte))).max():.1e}")
    ref_ridge = Ridge(alpha=tau_best).fit(Z, ytr)
    print(f"   Ridge vs sklearn Ridge: max|dw| = {np.abs(ridge.w_ - ref_ridge.coef_).max():.1e}, "
          f"max|dpred| = {np.abs(ridge.predict(Xte) - ref_ridge.predict(ridge.transform(Xte))).max():.1e}")
    ref_pcr = make_pipeline(PCA(n_components=k_best, svd_solver="full"), LinearRegression()).fit(Z, ytr)
    print(f"   PCR vs sklearn PCA+LinearRegression: max|dpred| = {np.abs(pcr.predict(Xte) - ref_pcr.predict(pcr.transform(Xte))).max():.1e}")


def synthetic_section():
    print("\nСинтетический пример: линейно зависимые признаки")
    rng = np.random.default_rng(SEED)
    mixing, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    base = rng.normal(size=(500, 3)) @ mixing.T
    X = np.c_[base + 0.05 * rng.normal(size=base.shape), base[:, 0] + base[:, 1]]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCASVD().fit(X)
    E = pca.residual_share()
    print(f"   {X.shape[1]} признаков порождены 3 скрытыми факторами, последний - точная сумма двух других (+ шум 0.05)")
    print("   собственные значения: " + " ".join(f"{v:.3f}" for v in pca.explained_variance_))
    print(f"   E_3 = {E[3]:.4f}; эффективная размерность (eps = 0.05): {effective_dimension(E, 0.05)}, крутой склон: {steep_slope(E)}")
    exact = np.c_[base, base[:, 0] + base[:, 1]]
    exact = (exact - exact.mean(axis=0)) / exact.std(axis=0)
    s = np.linalg.svd(exact, compute_uv=False)
    print(f"   без шума: сингулярные числа {np.array2string(s, precision=3, suppress_small=True)}, ранг {int(np.sum(s > 1e-8 * s[0]))}")
    print(f"   PCA по линейно зависимым признакам работает (SVD определено для любой матрицы), "
          f"восстановление по 3 осям: E_3 = {PCASVD().fit(exact).residual_share()[3]:.2e}")
    y = X @ rng.normal(size=X.shape[1]) + rng.normal(size=500)
    w = SVDRegression("ols").fit(X, y).w_
    print(f"   МНК через SVD с отбрасыванием нулевых сингулярных чисел: ||w|| = {np.linalg.norm(w):.3f}")


def main():
    print("=" * 70)
    print("Лабораторная работа №4: PCA через сингулярное разложение")
    print("=" * 70)

    data = {}
    for step, name in enumerate(("California Housing", "Diabetes"), 1):
        d = data[name] = load(name)
        print(f"\n{step}. Датасет {name}: train {d['X_train'].shape}, test {d['X_test'].shape}")
        print(f"   Признаки: {', '.join(d['features'])}")
        pca_section(d)
        regression_section(d)

    synthetic_section()

    plot_correlations(data, OUT / "pca_correlations.png")
    plot_spectrum(data, OUT / "pca_spectrum.png")
    plot_loadings(data, OUT / "pca_loadings.png")
    plot_projection(data, OUT / "pca_projection.png")
    plot_regression_k(data, OUT / "regression_pcr.png")
    plot_ridge(data, OUT / "regression_ridge.png")
    print("\nГрафики сохранены в директорию lab4.")


if __name__ == "__main__":
    main()
