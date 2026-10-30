# 1. выбрать датасет для линейной регрессии;
# 2. реализовать PCA через сингулярное разложение;
# 3. определить эффективную размерность выборки;
# 4. показать эквивалентность с эталонной реализацией;
# 5. подготовить отчет.

import os
import numpy as np
import pandas as pd
import pca
import plots
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
EPS = 0.05  # допустимая доля потерянной дисперсии для эффективной размерности

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def path(filename):
    return os.path.join(OUTPUT_DIR, filename)


# 1. выбрать датасет для линейной регрессии;
def load_data():
    data = fetch_california_housing()
    X, y = data.data, data.target  # y - медианная стоимость дома, 100 000 $
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, data.feature_names


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, names = load_data()
    n = X_train.shape[1]
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    # мультиколлинеарность: число обусловленности F^T F (слайд 9)
    lam_all = np.linalg.svd(X_train, compute_uv=False) ** 2
    print(f"\nμ(FᵀF) = λmax/λmin = {lam_all[0] / lam_all[-1]:.2f}")
    print("Корреляция признаков |r| > 0.7:")
    corr = np.corrcoef(X_train.T)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > 0.7:
                print(f"  {names[i]} - {names[j]}: {corr[i, j]:+.3f}")

    # 2. PCA через сингулярное разложение
    own = pca.PCA().fit(X_train)
    ref = SkPCA(svd_solver="full").fit(X_train)

    # 3. эффективная размерность
    print("\nСобственные значения λⱼ матрицы FᵀF и ошибка Eₘ:")
    print(" m |      λₘ | доля дисп. | накопл. |     Eₘ")
    for m in range(1, n + 1):
        print(f"{m:2d} | {own.eigvals[m - 1]:8.2f} | {own.eigvals[m - 1] / own.eigvals.sum():10.4f} | "
              f"{1 - own.error(m):7.4f} | {own.error(m):.4f}")
    m_eff = own.effective_dim(EPS)
    ratios = [own.error(m - 1) / own.error(m) for m in range(2, n)]  # критерий крутого склона: E_{m-1} >> E_m
    print(f"\nЭффективная размерность при ε = {EPS}: m = {m_eff}")
    print(f"Критерий крутого склона: max E_(m-1)/E_m = {max(ratios):.2f} при m = {2 + int(np.argmax(ratios))}")
    plots.plot_spectrum(own.eigvals, EPS, m_eff, path("spectrum.png"))

    # 4. эквивалентность с эталоном
    print("\nСравнение со sklearn.decomposition.PCA:")
    G_own, G_ref = own.transform(X_test), ref.transform(X_test)
    print(f"max |λ_own/(l-1) - explained_variance_| = {np.abs(own.eigvals / (own.l - 1) - ref.explained_variance_).max():.2e}")
    print(f"max |u_own - u_ref|  (компоненты, со знаком)    = {np.abs(own.components_ - ref.components_).max():.2e}")
    print(f"max ||u_own| - |u_ref||  (компоненты, до знака) = {np.abs(np.abs(own.components_) - np.abs(ref.components_)).max():.2e}")
    print(f"max |G_own - G_ref|  (test)                     = {np.abs(G_own - G_ref).max():.2e}")
    for m in (m_eff, n):
        o = pca.PCA(m).fit(X_train)
        r = SkPCA(m, svd_solver="full").fit(X_train)
        print(f"  m = {m}: max |восстановление_own - восстановление_ref| (test) = "
              f"{np.abs(o.inverse_transform(o.transform(X_test)) - r.inverse_transform(r.transform(X_test))).max():.2e}, "
              f"E_m own = {o.error(m):.6f}, E_m ref = {1 - r.explained_variance_ratio_.sum():.6f}")
    plots.plot_equivalence(own.eigvals / (own.l - 1), ref.explained_variance_, G_own, G_ref, path("equivalence.png"))

    # 4. показать эквивалентность с эталонной реализацией;
    full = LinearRegression().fit(X_train, y_train)
    rmse_full = rmse(y_test, full.predict(X_test))
    print(f"\nМНК на всех признаках: RMSE train = {rmse(y_train, full.predict(X_train)):.4f}, "
          f"test = {rmse_full:.4f}, R² test = {r2_score(y_test, full.predict(X_test)):.4f}")
    print(" m | RMSE train | RMSE test | R² test | max |pred_own - pred_sklearn| | ||α||")
    ms = list(range(1, n + 1))
    tr, te = [], []
    for m in ms:
        own_reg = pca.PCRegression(m).fit(X_train, y_train)
        ref_reg = make_pipeline(SkPCA(m, svd_solver="full"), LinearRegression()).fit(X_train, y_train)
        p_own, p_ref = own_reg.predict(X_test), ref_reg.predict(X_test)
        tr.append(rmse(y_train, own_reg.predict(X_train)))
        te.append(rmse(y_test, p_own))
        print(f"{m:2d} | {tr[-1]:10.4f} | {te[-1]:9.4f} | {r2_score(y_test, p_own):7.4f} | "
              f"{np.abs(p_own - p_ref).max():27.2e} | {np.linalg.norm(own_reg.alpha):.4f}")
    print(f"\nВеса МНК:  ||α|| = {np.linalg.norm(full.coef_):.4f}")
    plots.plot_regression(ms, tr, te, rmse_full, path("regression.png"))
