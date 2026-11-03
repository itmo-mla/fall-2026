import matplotlib.pyplot as plt
import numpy as np


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(str(filename), dpi=200)
    plt.close(fig)


def plot_correlations(data, filename):
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 5.4))
    for ax, (name, d) in zip(np.atleast_1d(axes), data.items()):
        C = np.corrcoef(d["X_train"], rowvar=False)
        im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        ticks = range(len(d["features"]))
        ax.set_xticks(ticks, d["features"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(ticks, d["features"], fontsize=8)
        for i in ticks:
            for j in ticks:
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(f"Корреляции признаков: {name}")
        fig.colorbar(im, ax=ax, label="коэффициент корреляции")
    _save(fig, filename)


def plot_spectrum(data, filename):
    fig, axes = plt.subplots(2, len(data), figsize=(6.2 * len(data), 8), squeeze=False)
    for col, (name, d) in enumerate(data.items()):
        lam, E, m = d["pca"].explained_variance_, d["E"], d["m_eff"]
        idx = np.arange(1, len(lam) + 1)
        ax = axes[0, col]
        ax.bar(idx, lam, color="tab:blue", label="дисперсия вдоль компоненты")
        ax.axhline(lam.mean(), color="tab:red", linestyle="--", label="среднее (порог Кайзера)")
        ax.set_xticks(idx)
        ax.set_xlabel("номер компоненты j")
        ax.set_ylabel("собственное значение (дисперсия)")
        ax.set_title(f"Спектр: {name}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.plot(range(len(E)), E, "o-", color="tab:green", label="E_m (потерянная доля)")
        for eps, style in ((0.05, "--"), (0.01, ":")):
            ax.axhline(eps, color="gray", linestyle=style, label=f"eps = {eps}")
        ax.axvline(m, color="tab:red", label=f"эффективная размерность m = {m} (eps = 0.05)")
        ax.set_xticks(range(len(E)))
        ax.set_xlabel("число компонент m")
        ax.set_ylabel("E_m")
        ax.set_title(f"Ошибка восстановления: {name}")
        ax.legend()
        ax.grid(alpha=0.3)
    _save(fig, filename)


def plot_loadings(data, filename):
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 5.2))
    for ax, (name, d) in zip(np.atleast_1d(axes), data.items()):
        V = d["pca"].components_
        im = ax.imshow(V.T, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(V.shape[0]), [str(j + 1) for j in range(V.shape[0])])
        ax.set_yticks(range(len(d["features"])), d["features"], fontsize=8)
        ax.set_xlabel("номер главной компоненты")
        ax.set_ylabel("исходный признак")
        ax.set_title(f"Нагрузки признаков на компоненты: {name}")
        fig.colorbar(im, ax=ax, label="координата главной оси")
    _save(fig, filename)


def plot_projection(data, filename):
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 5.2))
    for ax, (name, d) in zip(np.atleast_1d(axes), data.items()):
        Z = d["pca"].transform(d["X_train_std"], 2)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=d["y_train"], cmap="viridis", s=6, alpha=0.6)
        r = d["pca"].explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({r[0] * 100:.1f}% дисперсии)")
        ax.set_ylabel(f"PC2 ({r[1] * 100:.1f}% дисперсии)")
        ax.set_xlim(*np.percentile(Z[:, 0], [0.5, 99.5]))
        ax.set_ylim(*np.percentile(Z[:, 1], [0.5, 99.5]))
        ax.set_title(f"Проекция на PC1, PC2: {name}\n(оси обрезаны по 0.5-99.5 перцентилям)", fontsize=10)
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="значение ответа y")
    _save(fig, filename)


def plot_regression_k(data, filename):
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 4.8))
    for ax, (name, d) in zip(np.atleast_1d(axes), data.items()):
        ks = d["ks"]
        ax.plot(ks, d["pcr_cv"], "o-", label="PCR: R^2 (кросс-валидация)")
        ax.plot(ks, d["pcr_test"], "s--", label="PCR: R^2 (тест)")
        ax.axhline(d["ols_test"], color="gray", linestyle=":", label="МНК, все признаки: R^2 (тест)")
        ax.axvline(d["k_best"], color="tab:red", alpha=0.6, label=f"k* = {d['k_best']} (по CV)")
        ax.set_xticks(ks)
        ax.set_xlabel("число главных компонент k")
        ax.set_ylabel("R^2")
        ax.set_title(f"Регрессия на главных компонентах: {name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, filename)


def plot_ridge(data, filename):
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 4.8))
    for ax, (name, d) in zip(np.atleast_1d(axes), data.items()):
        ax.semilogx(d["taus"], d["ridge_cv"], "o-", label="Ridge: R^2 (кросс-валидация)")
        ax.axvline(d["tau_best"], color="tab:red", alpha=0.6, label=f"tau* = {d['tau_best']:.3g}")
        ax.axvline(d["tau_gcv"], color="tab:green", linestyle="--", alpha=0.8, label=f"tau по GCV = {d['tau_gcv']:.3g}")
        ax.axhline(d["ols_cv"], color="gray", linestyle=":", label="МНК (tau = 0): R^2 (CV)")
        ax.set_xlabel("коэффициент регуляризации tau")
        ax.set_ylabel("R^2 (кросс-валидация)")
        ax.set_title(f"Подбор tau для гребневой регрессии: {name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, filename)
