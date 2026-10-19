import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spectrum(eigenvalues, ratio, residual, path):
    m = np.arange(1, len(eigenvalues) + 1)
    fig, (ax_l, ax_e) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.bar(m, ratio, color="tab:blue")
    ax_l.plot(m, np.cumsum(ratio), color="tab:red", marker="o", label="накопленная доля")
    ax_l.set_xlabel("номер компоненты j")
    ax_l.set_ylabel("доля λ_j / Σλ")
    ax_l.set_title("Спектр матрицы FᵀF")
    ax_l.grid(alpha=0.3)
    ax_l.legend()

    ax_e.semilogy(np.arange(0, len(residual)), residual, marker="o")
    for eps in (0.15, 0.05, 0.01):
        ax_e.axhline(eps, color="gray", linestyle="--", linewidth=0.8)
        ax_e.text(len(residual) - 1, eps * 1.08, f"ε = {eps}", ha="right", fontsize=8)
    ax_e.set_xlabel("число компонент m")
    ax_e.set_ylabel("E_m (лог. шкала)")
    ax_e.set_title("Доля потерянной нормы E_m")
    ax_e.grid(alpha=0.3)
    save(fig, path)


def plot_components(components, feature_names, ratio, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(components.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ratio)))
    ax.set_xticklabels([f"{j + 1}\n{r:.0%}" for j, r in enumerate(ratio)])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    for (i, j), v in np.ndenumerate(components.T):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("главная ось (доля дисперсии)")
    ax.set_title("Нагрузки признаков на главные оси (столбцы U)")
    fig.colorbar(im, ax=ax)
    save(fig, path)


def plot_reconstruction(errors, tail_sums, path):
    m = np.arange(len(errors))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(m, errors, marker="o", label="||G Uᵀ − F||²")
    ax.semilogy(m, tail_sums, marker="x", linestyle="--", label="Σ_{j>m} λ_j")
    ax.set_xlabel("число компонент m")
    ax.set_ylabel("ошибка восстановления")
    ax.set_title("Теорема о PCA: ошибка равна сумме отброшенных λ")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_regression(ms, r2_train, r2_test, r2_ols, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ms, r2_train, marker="o", label="обучение")
    ax.plot(ms, r2_test, marker="o", label="тест")
    ax.axhline(r2_ols, color="gray", linestyle="--", label=f"МНК по всем признакам, тест {r2_ols:.3f}")
    ax.set_xlabel("число компонент m")
    ax.set_ylabel("R²")
    ax.set_title("Линейная регрессия на m главных компонентах")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)
