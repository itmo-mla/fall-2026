import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(newton_history, irls_history, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(len(newton_history)), newton_history, marker="o", label="Ньютон-Рафсон")
    plt.plot(np.arange(len(irls_history)), irls_history, marker="s", label="IRLS")
    plt.xlabel("Итерация")
    plt.ylabel("Функционал Q(w)")
    plt.title("Сходимость методов оптимизации")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probabilities(test_sigma, y_tilde_test, output_path):
    order = np.argsort(test_sigma)
    sorted_sigma = np.asarray(test_sigma)[order]
    sorted_labels = np.asarray(y_tilde_test)[order]

    colors = np.where(sorted_labels == 1, "tab:red", "tab:blue")

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(len(sorted_sigma)), sorted_sigma, c=colors, s=12, alpha=0.6)
    plt.xlabel("Объекты (отсортированы по вероятности)")
    plt.ylabel("sigma_i = P(y_i | x_i)")
    plt.title("Вероятности правильной классификации на тестовой выборке")
    plt.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="Класс -1"),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label="Класс +1"),
    ])
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
