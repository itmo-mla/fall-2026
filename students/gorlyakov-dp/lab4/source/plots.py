import matplotlib.pyplot as plt
import numpy as np


def plot_dimension_selection(singular_values, errors, epsilon, m, path):
    dimensions = np.arange(1, len(singular_values) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(dimensions, singular_values**2, marker="o")
    axes[0].set(
        xlabel="Номер компоненты j", ylabel="λ_j = s_j²",
        title="Собственные значения по компонентам",
    )

    axes[1].plot(np.arange(len(errors)), errors, marker="o")
    axes[1].axhline(epsilon, color="tab:red", linestyle="--", label=f"ε = {epsilon}")
    axes[1].scatter(m, errors[m], color="tab:red", zorder=3, label=f"По порогу: m = {m}")
    axes[1].set(xlabel="Число компонент m", ylabel="E_m", title="Ошибка восстановления")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(dimensions)
        ax.grid(alpha=0.3)
    axes[1].set_xticks(np.arange(len(errors)))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
