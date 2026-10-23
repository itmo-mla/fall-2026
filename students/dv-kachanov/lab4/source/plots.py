import numpy as np
import matplotlib.pyplot as plt


def plot_scree(lam, errors, m, eps, output_path):
    lam = np.asarray(lam, dtype=float)
    js = np.arange(1, len(lam) + 1)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(js, lam, marker="o", label="lambda_j")
    axes[0].axvline(m, linestyle="--", label=f"m = {m}")
    axes[0].set_xlabel("Номер компоненты j")
    axes[0].set_ylabel("Собственное значение lambda_j матрицы S = F^T F")
    axes[0].set_title("Спектр матрицы S (стандартизованные данные)")
    axes[0].grid()
    axes[0].legend()

    ms = np.arange(0, len(errors))
    axes[1].plot(ms, errors, marker="o", label="E_m")
    axes[1].axhline(eps, linestyle="--", label=f"epsilon = {eps}")
    axes[1].axvline(m, linestyle="--", label=f"m = {m}")
    axes[1].set_xlabel("Размерность m")
    axes[1].set_ylabel("Относительная ошибка E_m")
    axes[1].set_title("Относительная ошибка восстановления")
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reference_comparison(lam, lam_ref, errors, errors_ref, m, output_path):
    lam = np.asarray(lam, dtype=float)
    lam_ref = np.asarray(lam_ref, dtype=float)
    errors = np.asarray(errors, dtype=float)
    errors_ref = np.asarray(errors_ref, dtype=float)

    js = np.arange(1, len(lam) + 1)
    ms = np.arange(0, len(errors))

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(js, lam, marker="o", linestyle="-", label="наша реализация")
    axes[0].plot(js, lam_ref, marker="x", linestyle="--", label="sklearn PCA")
    axes[0].set_xlabel("Номер компоненты j")
    axes[0].set_ylabel("Собственное значение lambda_j")
    axes[0].set_title("Спектр S: наша реализация vs sklearn")
    axes[0].grid()
    axes[0].legend()
    axes[0].text(
        0.98,
        0.05,
        f"max |Δlambda| = {np.max(np.abs(lam - lam_ref)):.2e}",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    axes[1].plot(ms, errors, marker="o", linestyle="-", label="наша реализация")
    axes[1].plot(ms, errors_ref, marker="x", linestyle="--", label="sklearn PCA")
    axes[1].axvline(m, linestyle=":", label=f"m = {m}")
    axes[1].set_xlabel("Размерность m")
    axes[1].set_ylabel("Относительная ошибка E_m")
    axes[1].set_title("Ошибка восстановления: наша реализация vs sklearn")
    axes[1].grid()
    axes[1].legend()
    axes[1].text(
        0.98,
        0.05,
        f"max |ΔE_m| = {np.max(np.abs(errors - errors_ref)):.2e}",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_projection(G, y, output_path):
    G = np.asarray(G, dtype=float)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(G[:, 0], G[:, 1], c=y, s=35, alpha=0.6)
    plt.colorbar(scatter, label="Прогрессирование диабета")
    plt.xlabel("Первый главный признак g_1 (первый столбец G = F U)")
    plt.ylabel("Второй главный признак g_2 (второй столбец G = F U)")
    plt.title("Проекция обучающей выборки: первые два столбца G")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction(errors, m, output_path):
    ms = np.arange(0, len(errors))
    plt.figure(figsize=(9, 5))
    plt.plot(ms, errors, marker="o", label="E_m = ||G U^T - F||^2 / ||F||^2")
    plt.axvline(m, linestyle="--", label=f"m = {m}")
    plt.xlabel("Размерность m")
    plt.ylabel("Относительная ошибка E_m")
    plt.title("Ошибка восстановления (стандартизованные данные)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
