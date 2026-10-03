import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import TARGET
from lin_clf import LinearClassifier


def visualise_marg(X, y,mode = "corr") -> None:
    model = LinearClassifier()
    if mode == "corr":
        model.corr_init(X, y)
    elif mode == "train":
        model.fit(X,y,n_iter=1000, batch_size=264, momentum=0.8, verbose=300, sampling= "margin", tao = 0.8)
    elif mode == "random":
        model.random_init(X)
    else:
        raise ValueError("corr или train или random")
    marg = model.margin(X, y)
    plt.scatter(np.arange(len(marg)), np.sort(marg))
    plt.xlabel("Объект")
    plt.ylabel("Отступ")
    plt.title(f"Отступ объектов с {mode}", )

    plt.show()



def visualise_corr(data:pd.DataFrame):
    data_without_target = data.drop(columns=TARGET)
    corr_mat = data_without_target.corr(numeric_only=True)
    n = len(corr_mat)
    fig, ax = plt.subplots(figsize = (10,8))
    im = ax.imshow(corr_mat, cmap='coolwarm', vmin = -1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_mat.columns, rotation = 90)
    ax.set_yticklabels(corr_mat.columns)
    fig.colorbar(im, ax = ax)
    plt.tight_layout()
    plt.title("Матрица корреляций")

    for i in range(n):
        for j in range(n):
            ax.text(j,i, f"{corr_mat.iloc[i,j]:.2f}", ha = 'center', va = "center", color = 'black', fontsize = 9)


    plt.show()
