"""Визуализация экспериментов лабораторной работы № 4."""

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_test, models, output_path):
    """Сравнить предсказания четырёх моделей на тестовой выборке."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Сортируем по реальному y, а предсказания переставляем в том же порядке.
    order = np.argsort(y_test)
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        ax.plot(y_test[order], color="black", linewidth=2, label="Реальные продажи", zorder=5)
        for i, (name, predictions) in enumerate(models):
            ax.plot(predictions[order], linestyle="--" if i % 2 else "-",
                    linewidth=1, alpha=0.7, label=name)
        ax.set_title("Предсказания на тестовой выборке")
        ax.set_xlabel("Номер объекта после сортировки по реальным продажам")
        ax.set_ylabel("Продажи")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def plot_reconstruction_error(errors, output_path):
    """Сохранить график ошибки восстановления в зависимости от размерности."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    try:
        ax.plot(range(1, len(errors) + 1), errors, marker="o", label=r"$E_m$")
        ax.set_xlabel("m")
        ax.set_ylabel(r"$E_m$")
        ax.grid(True)
        ax.legend()
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)

