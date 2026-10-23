import matplotlib.pyplot as plt
import numpy as np

from lab4.dataset import download_dataset, preprocess_data
from lab4.PCA import PCA


def draw_e_plot(e_list, elbow_idx=1):
    x = np.arange(len(e_list))
    y = np.array(e_list)

    plt.plot(x, y, "-", color="steelblue", label="E_m")
    plt.scatter(x, y, color="blue", zorder=3, label="points")
    plt.scatter(x[elbow_idx], y[elbow_idx], color="red", s=100, zorder=4, label="elbow")

    plt.xlabel("m (components kept)")
    plt.ylabel("E_m")
    plt.legend()
    plt.grid(True)


def main():
    df = download_dataset()
    x = preprocess_data(df)

    pca = PCA(x)
    lam = pca.s**2
    total = lam.sum()

    # E_m for m = 0..n
    E = [lam[m:].sum() / total for m in range(len(lam) + 1)]

    draw_e_plot(E)
    plt.savefig("./images/e_plot.png")
    plt.show()
