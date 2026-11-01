import matplotlib.pyplot as plt


def plot_loo_risk(ks, risks, k, path):
    best_risk = risks[ks == k][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, limit in zip(axes, [ks[-1], 50]):
        mask = ks <= limit
        ax.plot(ks[mask], risks[mask], linewidth=1.5)
        ax.scatter(k, best_risk, color="tab:red", zorder=3, label=f"Минимум, k = {k}")
        ax.set(xlabel="k", ylabel="Доля ошибок LOO", title=f"k от 1 до {limit}")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
