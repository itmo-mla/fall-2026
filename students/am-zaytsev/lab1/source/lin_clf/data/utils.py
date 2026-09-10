import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_corr(df):
    # 2. Calculate the correlation matrix
    corr_matrix = df.corr()

    # 3. Create a mask to hide the upper triangle (since correlation is symmetric)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 4. Set up the matplotlib figure
    plt.figure(figsize=(14, 11))

    # 5. Draw the heatmap with the mask and a nice color palette
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        annot=True,  # Set to True if you want to see the numbers, but it will be crowded
    )

    plt.title("Correlation Matrix - Breast Cancer Dataset", fontsize=16)
    plt.tight_layout()
    return plt


def prepocess(data):
    print("Preprocess:", data)
