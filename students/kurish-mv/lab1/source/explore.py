import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import COLUMNS, TARGET, ZERO_AS_MISSING


def explore(df, out_dir):
    print(df.head())
    df.info()
    print(df.describe())
    print("\nНулей в признаках:\n", (df[ZERO_AS_MISSING] == 0).sum())
    print("\nКлассы:\n", df[TARGET].value_counts())

    clean = df.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)

    plt.figure(figsize=(9, 7))
    sns.heatmap(clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Корреляционная матрица")
    plt.tight_layout()
    plt.savefig(out_dir / "01_corr_matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), COLUMNS[:-1]):
        for c in (0, 1):
            ax.hist(clean.loc[clean[TARGET] == c, col].dropna(), bins=20,
                    alpha=0.6, density=True, label=f"class {c}")
        ax.set_title(col)
    axes.ravel()[0].legend()
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "02_features_by_class.png", dpi=150)
    plt.close()
