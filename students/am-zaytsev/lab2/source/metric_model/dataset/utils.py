import os

import kagglehub
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_encode(row: np.array):
    row_2d = row.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform(row_2d)
    return one_hot


def remove_row(df, row_i):
    mask = np.ones(df.shape[0])
    mask[row_i] = 0
    mask = mask == 1
    return df[mask]


def draw_data(df: pd.DataFrame, target_name: str):
    # Features and labels
    features = [i for i in list(df) if i != target_name]
    X = df[features].values
    y = df[target_name].values

    # Scale features (important for t-SNE)
    X_scaled = StandardScaler().fit_transform(X)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # try 5–50; ~sqrt(n) is a good start
        learning_rate="auto",
        init="pca",
        random_state=42,
        # n_iter=1000
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # Plot
    plt.figure(figsize=(9, 7))
    for label in sorted(df[target_name].unique()):
        mask = y == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], s=15, alpha=0.7, label=f"Class {label}"
        )

    plt.title("t-SNE Visualization of Classes")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.tight_layout()
