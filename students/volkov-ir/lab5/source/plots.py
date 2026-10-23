import matplotlib.pyplot as plt


def plot_q_history(Q_newton, Q_irls, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(len(Q_newton)), Q_newton, marker="o", color="tab:blue", label="Ньютон-Рафсон, старт из нулей")
    plt.plot(range(len(Q_irls)), Q_irls, marker="s", ls="--", color="tab:orange", label="IRLS, старт из МНК")
    plt.xlabel("номер итерации (0 - начальные веса)")
    plt.ylabel("Q на train")
    plt.title("Функция потерь по итерациям")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
