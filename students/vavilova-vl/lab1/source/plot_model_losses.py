import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = PROJECT_DIR / "models" / "training_loss_history.csv"
OUTPUT_FILE = PROJECT_DIR / "graphs" / "loss_comparison.png"


def main():
    histories = defaultdict(list)
    with HISTORY_FILE.open(encoding="utf-8-sig", newline="") as history_file:
        for row in csv.DictReader(history_file):
            histories[row["model"]].append(float(row["train_half_mse"]))

    figure, axis = plt.subplots(figsize=(12, 7))
    for model_name, losses in histories.items():
        progress = [
            100 * step / (len(losses) - 1) if len(losses) > 1 else 100
            for step in range(len(losses))
        ]
        axis.plot(progress, losses, label=model_name, linewidth=1.8)

    axis.set_xlabel("Ход обучения, %")
    axis.set_ylabel("Train half-MSE")
    axis.set_title("Изменение функции потерь моделей")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    print(f"График сохранен: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()