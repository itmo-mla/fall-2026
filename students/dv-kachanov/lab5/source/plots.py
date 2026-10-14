import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(newton_history, irls_history, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(len(newton_history)), newton_history, marker="o", label="Ньютон-Рафсон")
    plt.plot(np.arange(len(irls_history)), irls_history, marker="s", label="IRLS")
    plt.xlabel("Итерация")
    plt.ylabel("Регуляризованный функционал J(w)")
    plt.title("Сходимость методов оптимизации")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probabilities(test_proba, y_test, output_path):
    order = np.argsort(test_proba)
    sorted_proba = test_proba[order]
    sorted_labels = y_test[order]

    colors = np.where(sorted_labels == 1, "tab:red", "tab:blue")

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(len(sorted_proba)), sorted_proba, c=colors, s=12, alpha=0.6)
    plt.axhline(0.5, linestyle="--", label="Порог 0.5")
    plt.xlabel("Объекты (отсортированы по вероятности)")
    plt.ylabel("P(y = 1 | x)")
    plt.title("Предсказанные вероятности на тестовой выборке")
    plt.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="Класс 0"),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label="Класс 1"),
        plt.Line2D([], [], linestyle="--", color="black", label="Порог 0.5"),
    ])
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
