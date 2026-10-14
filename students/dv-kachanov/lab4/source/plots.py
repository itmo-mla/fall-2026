import numpy as np
import matplotlib.pyplot as plt


def plot_scree(eigenvalues, ratios, effective_k, output_path):
    components = np.arange(1, len(eigenvalues) + 1)
    cumulative = np.cumsum(ratios)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(components, eigenvalues, marker="o", label="lambda_j")
    axes[0].axhline(eigenvalues.mean(), linestyle="--", label="Среднее lambda (Кайзер)")
    axes[0].axvline(effective_k, linestyle="--", label=f"k* = {effective_k}")
    axes[0].set_xlabel("Номер компоненты j")
    axes[0].set_ylabel("Собственное значение lambda_j")
    axes[0].set_title("Скри-график собственных значений")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(components, cumulative, marker="o", label="Накопленная доля дисперсии")
    axes[1].axhline(0.95, linestyle="--", label="Порог 0.95")
    axes[1].axvline(effective_k, linestyle="--", label=f"k* = {effective_k}")
    axes[1].set_xlabel("Число компонент k")
    axes[1].set_ylabel("Доля объясненной дисперсии")
    axes[1].set_title("Накопленная объясненная дисперсия")
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_projection(Z, y, output_path):
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=35, alpha=0.6)
    plt.colorbar(scatter, label="Прогрессирование диабета")
    plt.xlabel("Первая главная компонента")
    plt.ylabel("Вторая главная компонента")
    plt.title("Проекция обучающей выборки на две главные компоненты")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction(errors, effective_k, output_path):
    components = np.arange(1, len(errors) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(components, errors, marker="o", label="MSE восстановления")
    plt.axvline(effective_k, linestyle="--", label=f"k* = {effective_k}")
    plt.xlabel("Число компонент k")
    plt.ylabel("MSE восстановления")
    plt.title("Ошибка восстановления в зависимости от размерности")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
