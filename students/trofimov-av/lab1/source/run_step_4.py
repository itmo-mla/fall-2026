from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from margins import centroid_linear_rule, decision_function
from quality import recursive_quality_history


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    weights, bias = centroid_linear_rule(dataset.features, dataset.labels)
    scores = decision_function(dataset.features, weights, bias)
    object_losses = 0.5 * (scores - dataset.labels) ** 2
    exact_quality = float(np.mean(object_losses))

    rng = np.random.default_rng(42)
    iterations = len(object_losses) * 20
    indices = np.concatenate(
        [rng.permutation(len(object_losses)) for _ in range(20)]
    )
    observed_losses = object_losses[indices]
    smoothing = 0.02
    history = recursive_quality_history(
        observed_losses,
        smoothing=smoothing,
        initial_quality=float(observed_losses[0]),
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        np.arange(1, iterations + 1),
        history,
        label="Рекуррентная оценка",
        color="tab:blue",
    )
    axis.axhline(
        exact_quality,
        color="tab:red",
        linestyle="--",
        label="Точное среднее по выборке",
    )
    axis.set_title("Рекуррентная оценка функционала качества")
    axis.set_xlabel("Номер предъявленного объекта")
    axis.set_ylabel("Квадратичная потеря")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "quality_estimate.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Точное значение Q: {exact_quality:.6f}")
    print(f"Конечная рекуррентная оценка: {history[-1]:.6f}")
    print(f"Абсолютное отклонение: {abs(history[-1] - exact_quality):.6f}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
