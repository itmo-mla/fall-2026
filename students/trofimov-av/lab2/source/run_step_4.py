from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_split, standardize
from validation import select_k_loo


def main():
    x_train, x_test, y_train, _, _, _ = load_iris_split()
    x_train, _, _, _ = standardize(x_train, x_test)
    k_values = np.arange(1, 31)
    best_k, risks = select_k_loo(x_train, y_train, k_values)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(k_values, risks, marker="o", label="LOO-риск")
    axis.scatter(
        best_k,
        risks[best_k - 1],
        color="red",
        s=80,
        zorder=3,
        label=f"Выбрано k={best_k}",
    )
    axis.axhline(
        risks.min(),
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"Минимум={risks.min():.4f}",
    )
    axis.set_title("LOO-риск для разных значений k")
    axis.set_xlabel("Количество соседей k")
    axis.set_ylabel("Доля ошибок")
    axis.set_xticks(np.arange(1, 31, 2))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "loo_risk.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Выбрано k: {best_k}")
    print(f"Минимальный LOO-риск: {risks.min():.4f}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
