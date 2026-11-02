from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import data_load, split_and_scale, FEATURE_NAMES
from pca_svd import PCA_SVD
from dimensionality import dimension_table, choose_m_star, bonus_regression_table
from compare_sklearn import compare_with_sklearn

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

COLOR_MINE = "#4472C4"
COLOR_SKL = "#ED7D31"


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


X, y = data_load()
X_train, X_test, y_train, y_test = split_and_scale(X, y)
d = X_train.shape[1]

feature_target_corr = X.corrwith(pd.Series(y)).sort_values(key=np.abs, ascending=False)
print("корреляция признаков с прочностью:")
for name, c in feature_target_corr.items():
    print(f"  {name}: {c:.3f}")
print(f"  медиана прочности: {np.median(y):.2f}, среднее: {y.mean():.2f}")

plt.figure(figsize=(6, 4))
plt.hist(y, bins=30, color=COLOR_MINE, edgecolor="black")
plt.xlabel("прочность, МПа")
plt.ylabel("количество объектов")
plt.title("Распределение прочности бетона")
savefig("00_target_distribution.png")

plt.figure(figsize=(7, 6))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            cbar_kws={"label": "коэффициент корреляции"})
plt.title("Корреляция признаков состава бетона")
savefig("01_corr_heatmap.png")

plt.figure(figsize=(8, 5))
sns.boxplot(data=X, color=COLOR_MINE)
plt.xticks(rotation=30, ha="right")
plt.ylabel("значение признака")
plt.title("Масштабы признаков до стандартизации")
savefig("02_feature_scales.png")

pca = PCA_SVD().fit(X_train)

ortho_resid = np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(d)))
X_approx_full = pca.inverse_transform(pca.transform(X_train))
full_recon_resid = np.max(np.abs(X_approx_full - X_train))
ratio_sum = pca.explained_variance_ratio_.sum()

print("sanity-check PCA:")
print(f"  ортонормированность компонент, max|V@V.T - I| = {ortho_resid:.2e}")
print(f"  восстановление при m=d, max|X_approx - X| = {full_recon_resid:.2e}")
print(f"  сумма explained_variance_ratio = {ratio_sum:.10f}")

kaiser_m = int(np.sum(pca.explained_variance_ > 1))
print(f"\nправило Кайзера (lambda>1): m={kaiser_m}, "
      f"cum_var={np.cumsum(pca.explained_variance_ratio_)[kaiser_m - 1]:.3f}")

print("нагрузки PC1 (по убыванию модуля):")
for name, w in sorted(zip(FEATURE_NAMES, pca.components_[0]), key=lambda t: -abs(t[1])):
    print(f"  {name}: {w:.3f}")

Z_train_full = pca.transform(X_train)
print(f"корреляция PC1 с прочностью (train): {np.corrcoef(Z_train_full[:, 0], y_train)[0, 1]:.3f}")
print(f"корреляция PC2 с прочностью (train): {np.corrcoef(Z_train_full[:, 1], y_train)[0, 1]:.3f}")

table = dimension_table(pca, X_train, X_test)
cum_var = np.cumsum(pca.explained_variance_ratio_)
m_star = choose_m_star(cum_var, threshold=0.95)

print("\nразмерность vs качество:")
for row in table:
    print(f"  m={row['m']}  cum_var={row['cum_var']:.4f}  "
          f"mse_train={row['mse_train']:.4f}  mse_test={row['mse_test']:.4f}")
print(f"  выбранное m* (порог 95% дисперсии) = {m_star}")

fig, ax1 = plt.subplots(figsize=(7, 5))
idx = np.arange(1, d + 1)
ax1.bar(idx, pca.explained_variance_, color=COLOR_MINE)
ax1.set_xlabel("номер компоненты")
ax1.set_ylabel("собственное значение", color=COLOR_MINE)
ax2 = ax1.twinx()
ax2.plot(idx, cum_var * 100, marker="o", color=COLOR_SKL)
ax2.set_ylabel("накопленная доля дисперсии, %", color=COLOR_SKL)
ax2.axhline(90, color="gray", linestyle="--", linewidth=1)
ax2.axhline(95, color="gray", linestyle="--", linewidth=1)
ax2.axvline(m_star, color="gray", linestyle="--", linewidth=1)
ax2.annotate(f"m*={m_star}", (m_star, 50), xytext=(m_star + 0.1, 50))
plt.title("Scree-график и накопленная доля дисперсии")
savefig("03_scree_cumvar.png")

plt.figure(figsize=(7, 5))
ms = [row["m"] for row in table]
plt.plot(ms, [row["mse_train"] for row in table], color=COLOR_MINE, label="train")
plt.plot(ms, [row["mse_test"] for row in table], color=COLOR_SKL, linestyle="--", label="test")
plt.axvline(m_star, color="gray", linestyle="--", linewidth=1)
plt.xlabel("число компонент m")
plt.ylabel("MSE восстановления")
plt.legend()
plt.title("Ошибка восстановления от числа компонент")
savefig("04_reconstruction_error.png")

bonus = bonus_regression_table(pca, X_train, y_train, X_test, y_test, ms=[2, m_star, d])
print("\nбонус: R2 регрессии на m главных компонентах (test):")
for row in bonus:
    print(f"  m={row['m']}  R2={row['r2_test']:.4f}")

skl, signs, eq_rows, Z_mine, Z_skl, max_z_diff, max_x_diff = compare_with_sklearn(pca, X_train, svd_solver="auto")
skl_full, signs_full, eq_rows_full, _, _, max_z_diff_full, max_x_diff_full = compare_with_sklearn(
    pca, X_train, svd_solver="full"
)

print(f"\nsklearn.decomposition.PCA(svd_solver='auto') выбрал решатель: {skl._fit_svd_solver}")
print("эквивалентность со sklearn (auto -> " + skl._fit_svd_solver + "):")
for row in eq_rows:
    print(f"  k={row['component']}  |cos|={row['cos']:.6f}  "
          f"delta_var_ratio={row['var_ratio_diff']:.2e}")
print(f"  max|Z_mine - Z_sklearn (после выравнивания знака)| = {max_z_diff:.2e}")
print(f"  max|X_approx_mine - X_approx_sklearn| (m=d) = {max_x_diff:.2e}")

max_var_diff_full = max(row["var_ratio_diff"] for row in eq_rows_full)
max_var_diff_auto = max(row["var_ratio_diff"] for row in eq_rows)
print("\nэквивалентность со sklearn (svd_solver='full'):")
print(f"  max delta_var_ratio = {max_var_diff_full:.2e}")
print(f"  max|Z_mine - Z_sklearn| = {max_z_diff_full:.2e}")
print(f"  max|X_approx_mine - X_approx_sklearn| = {max_x_diff_full:.2e}")

plt.figure(figsize=(7, 5))
width = 0.35
plt.bar(idx - width / 2, pca.explained_variance_ratio_, width, color=COLOR_MINE, label="своя реализация")
plt.bar(idx + width / 2, skl.explained_variance_ratio_, width, color=COLOR_SKL, label="sklearn")
plt.xlabel("номер компоненты")
plt.ylabel("доля объяснённой дисперсии")
plt.legend()
plt.title("Доля объяснённой дисперсии: своя реализация vs sklearn")
savefig("05_compare_explained_variance.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
sc0 = axes[0].scatter(Z_mine[:, 0], Z_mine[:, 1], c=y_train, cmap="viridis", s=14)
axes[0].set_title("своя реализация")
axes[0].set_xlabel("1-я главная компонента")
axes[0].set_ylabel("2-я главная компонента")
axes[1].scatter(Z_skl[:, 0], Z_skl[:, 1], c=y_train, cmap="viridis", s=14)
axes[1].set_title("sklearn")
axes[1].set_xlabel("1-я главная компонента")
fig.colorbar(sc0, ax=axes, label="прочность, МПа")
fig.suptitle("Проекция на первые 2 компоненты: своя реализация vs sklearn")
plt.savefig(PLOTS_DIR / "06_compare_pc_scatter.png")
plt.close()

print(f"\nn={X.shape[0]}, train={len(y_train)}, test={len(y_test)}, признаков={d}")
