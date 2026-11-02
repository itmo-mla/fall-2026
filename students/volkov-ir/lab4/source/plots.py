import matplotlib.pyplot as plt


def plot_spectrum(lam, path):
    n = len(lam)
    shares = lam / lam.sum()

    plt.figure(figsize=(7, 4.5))
    plt.bar(range(1, n + 1), lam, color="tab:blue")
    # подпись доли над каждым столбиком
    for j in range(n):
        plt.text(j + 1, lam[j], str(round(shares[j] * 100, 1)) + " %", ha="center", va="bottom")
    plt.xlabel("номер направления")
    plt.ylabel("lambda")
    plt.title("Собственные числа lambda и их доля от суммы")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_e_m(e_m, path):
    n = len(e_m) - 1

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(0, n + 1), e_m, marker="o", color="black")
    plt.xlabel("число оставленных направлений m")
    plt.ylabel("потеря E_m")
    plt.title("Потеря информации E_m")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_rmse(rmse_list, rmse_full, path):
    n = len(rmse_list)

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, n + 1), rmse_list, marker="o", color="black", label="регрессия на m новых признаках")
    # обычная регрессия на всех исходных признаках
    plt.axhline(rmse_full, color="gray", ls="--", label="LinearRegression на исходных признаках")
    plt.xlabel("число новых признаков m")
    plt.ylabel("RMSE на test")
    plt.title("Ошибка прогноза mpg в зависимости от m")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
