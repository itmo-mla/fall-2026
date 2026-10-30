import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(eigvals, eps, m_eff, path):
    n = len(eigvals)
    ms = np.arange(0, n + 1)
    E = np.array([eigvals[m:].sum() / eigvals.sum() for m in ms])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(np.arange(1, n + 1), eigvals / eigvals.sum(), color="tab:blue")
    a1.set_xlabel("номер компоненты j")
    a1.set_ylabel("λⱼ / Σλ")
    a1.set_title("Доля дисперсии по компонентам")

    a2.plot(ms, E, "o-", color="tab:red")
    a2.axhline(eps, color="k", ls="--", lw=1, label=f"ε = {eps}")
    a2.axvline(m_eff, color="tab:green", ls=":", lw=1.5, label=f"m = {m_eff}")
    a2.set_xlabel("число компонент m")
    a2.set_ylabel("Eₘ")
    a2.set_title("Ошибка аппроксимации Eₘ")
    a2.legend()
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equivalence(var_own, var_ref, G_own, G_ref, path):
    n = len(var_own)
    j = np.arange(1, n + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    a1.bar(j - 0.2, var_ref, width=0.4, label="sklearn")
    a1.bar(j + 0.2, var_own, width=0.4, label="своя реализация")
    a1.set_xlabel("номер компоненты j")
    a1.set_ylabel("λⱼ / (ℓ − 1)")
    a1.set_title("Дисперсия вдоль компонент")
    a1.legend()
    a1.grid(alpha=0.3)

    a2.scatter(G_ref[:, 0], G_ref[:, 1], s=8, alpha=0.4, label="sklearn")
    a2.scatter(G_own[:, 0], G_own[:, 1], s=14, alpha=0.5, marker="x", label="своя реализация")
    lo, hi = np.percentile(G_ref[:, :2], [0.5, 99.5], axis=0)
    a2.set_xlim(lo[0], hi[0])
    a2.set_ylim(lo[1], hi[1])
    a2.set_xlabel("PC1")
    a2.set_ylabel("PC2")
    a2.set_title("Проекция test-выборки на PC1, PC2")
    a2.legend(markerscale=1.5)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regression(ms, rmse_train, rmse_test, rmse_full, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ms, rmse_train, "o-", label="train")
    ax.plot(ms, rmse_test, "s-", label="test")
    ax.axhline(rmse_full, color="k", ls="--", lw=1, label=f"МНК на всех признаках (test) = {rmse_full:.4f}")
    ax.set_xlabel("число главных компонент m")
    ax.set_ylabel("RMSE")
    ax.set_title("Регрессия на главных компонентах")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
