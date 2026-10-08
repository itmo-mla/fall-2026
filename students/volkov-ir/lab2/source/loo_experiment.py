from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import X_train, y_train, calculate_loo

IMAGES_DIR = Path(__file__).parent.parent / "images"
# для Парзена нужен (k+1)-й сосед среди ℓ-1 объектов, поэтому k ≤ ℓ-2
K_VALUES = list(range(1, len(X_train) - 1))
K_ZOOM = 50
METHODS = {"knn": ("KNN", "#2a78d6"), "parzen": ("окно Парзена, сумма по k соседям", "#eb6834"), "parzen_lecture": ("окно Парзена, сумма по всей выборке (лекция)", "#1baf7a")}


def select_k(k_values, loss):
    # при равной ошибке берём наибольшее k: больше соседей — устойчивее к шуму
    best = np.min(loss)
    return max(k for k, l in zip(k_values, loss) if l == best)


def plot_risk(results, best_k):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    for ax, k_max, title in [(axes[0], K_VALUES[-1], "все k"), (axes[1], K_ZOOM, f"k от 1 до {K_ZOOM}")]:
        for method, (label, color) in METHODS.items():
            ks = [k for k in K_VALUES if k <= k_max]
            loss = np.array(results[method][:len(ks)]) * 100
            ax.plot(ks, loss, color=color, linewidth=2, label=f"{label} (k* = {best_k[method]})")
            k_star = best_k[method]
            ax.scatter([k_star], [results[method][k_star - 1] * 100], s=64, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.set_title(f"Эмпирический риск LOO, {title}")
        ax.set_xlabel("k — число соседей")
        ax.set_ylabel("ошибка LOO, %")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / "loo_risk.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = {method: calculate_loo(X_train, y_train, K_VALUES, method) for method in METHODS}
    best_k = {method: select_k(K_VALUES, loss) for method, loss in results.items()}
    for method, (label, _) in METHODS.items():
        loss = np.array(results[method])
        minima = [k for k, l in zip(K_VALUES, loss) if l == loss.min()]
        print(f"{label}: min LOO = {loss.min():.4f} ({round(loss.min() * len(X_train))} ошибки из {len(X_train)}) при k = {minima}, выбрано k* = {best_k[method]}")
    plot_risk(results, best_k)
    print("график:", IMAGES_DIR / "loo_risk.png")
