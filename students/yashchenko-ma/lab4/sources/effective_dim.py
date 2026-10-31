"""Эффективная размерность выборки (лекция, слайд 23) + корреляции и обусловленность (слайды 9-10).

E_m = ||G U^T - F||^2 / ||F||^2 = (lambda_{m+1}+...+lambda_n) / (lambda_1+...+lambda_n).
Эффективная размерность - наименьшее целое m, при котором E_m <= eps.
Критерий "крутого склона": m, где E_{m-1} >> E_m.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from common import (get_regression_data, setup_style, save_fig,
                    C1, C2, C3, INK2, MUTED)
from pca_svd import PCASVD


def E_curve(eigvals):
    """E_m для m = 0..n (E_0 = 1)."""
    lam = np.asarray(eigvals, float)
    return np.concatenate([[1.0], 1.0 - np.cumsum(lam) / lam.sum()])


def effective_dim(eigvals, eps):
    E = E_curve(eigvals)
    return int(np.argmax(E <= eps))                 # первый m с E_m <= eps


def steep_slope_m(eigvals):
    """m = argmax E_{m-1}/E_m по m = 1..n-1 (слайд 23)."""
    E = E_curve(eigvals)
    n = len(E) - 1
    ratios = {m: E[m - 1] / E[m] for m in range(1, n) if E[m] > 0}
    return max(ratios, key=ratios.get), ratios


def main():
    setup_style()
    X_tr, _, _, _, names = get_regression_data()
    pca = PCASVD().fit(X_tr)
    lam = pca.eigvals_
    n = len(lam)

    # --- корреляции и обусловленность (слайды 9-10) ---
    corr = np.corrcoef(X_tr.T)
    cond = lam[0] / lam[-1]
    print("\nПризнаки:", names)
    print("Корреляционная матрица (train):\n", np.round(corr, 3))
    print("lambda_j (F^T F):", np.round(lam, 2))
    print("lambda_j / l (дисперсии ГК):", np.round(lam / len(X_tr), 4))
    print(f"mu(F^T F) = lambda_max/lambda_min = {cond:.2f}")

    # --- эффективная размерность ---
    E = E_curve(lam)
    print("\nm : E_m")
    for m in range(n + 1):
        print(f"{m} : {E[m]:.4f}")
    for eps in (0.20, 0.10, 0.05, 0.01):
        print(f"eps={eps:<5} -> m = {effective_dim(lam, eps)}")
    m_star, ratios = steep_slope_m(lam)
    print("E_{m-1}/E_m:", {k: round(v, 2) for k, v in ratios.items()})
    print(f"Критерий крутого склона: m = {m_star}")

    # --- график 1: корреляции ---
    cmap = LinearSegmentedColormap.from_list("div", [C1, "#f0efec", C2])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.grid(False)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), names, rotation=40, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="#0b0b0b")
    ax.set_title("Корреляции признаков (train, стандартизованные)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, "corr_heatmap.png")

    # --- график 2: scree + E_m ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(range(1, n + 1), lam / len(X_tr), "o-", color=C1, label="λ_j / ℓ")
    a1.set_xlabel("номер компоненты j")
    a1.set_ylabel("доля дисперсии λ_j / ℓ")
    a1.set_title("Спектр (scree plot)")
    a1.legend()

    a2.plot(range(0, n + 1), E, "o-", color=C1, label="E_m")
    for eps, col, ls in ((0.10, C2, "--"), (0.01, C3, ":")):
        a2.axhline(eps, color=col, ls=ls, lw=1.6, label=f"ε = {eps}")
    a2.axvline(m_star, color=MUTED, ls="-.", lw=1.2, label=f"крутой склон: m = {m_star}")
    a2.set_xlabel("m (число главных компонент)")
    a2.set_ylabel("E_m")
    a2.set_title("Эффективная размерность")
    a2.legend()
    save_fig(fig, "effective_dim.png")


if __name__ == "__main__":
    main()
