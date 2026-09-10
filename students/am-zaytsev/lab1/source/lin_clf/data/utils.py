import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data():
    # 1. Load the Breast Cancer dataset into a Pandas DataFrame
    cancer = load_breast_cancer()
    df_x = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_y = pd.DataFrame(cancer.target)
    return df_x, df_y


def _remove_corr_columns(df):
    for i in [
        "mean texture",
        "mean radius",
        "mean perimeter",
        "mean area",
        "worst radius",
        "worst perimeter",
        "perimeter error",
        "radius error",
        "mean concavity",
        "mean concave points",
        "worst compactness",
    ]:
        del df[i]
    return df


def _split(df_x, df_y):
    X_train, X_test, y_train, y_test = train_test_split(
        df_x,
        df_y,
        test_size=0.2,  # 20% of data goes to the test set, 80% to train
        random_state=42,  # Fixes the random seed so your results are reproducible
        stratify=df_y,
    )
    return X_train, X_test, y_train, y_test


def normalize(train_split, test_split):
    scaler = MinMaxScaler()
    scaler.set_output(
        transform="pandas"
    )  # Магия! Метод возвращает DataFrame вместо массива NumPy

    train_split = scaler.fit_transform(train_split)
    test_split = scaler.transform(test_split)
    return (train_split, test_split)


def preprocess(df_x, df_y):
    df_y = df_y * 2 - 1
    df_x = _remove_corr_columns(df_x)
    X_train, X_test, y_train, y_test = _split(df_x, df_y)
    X_train, X_test = normalize(X_train, X_test)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def fetch_batch(x, y, batch_size):
    batch_idx = np.random.choice(x.shape[0], batch_size, x.shape[0] <= batch_size)
    return x[batch_idx], y[batch_idx]


def draw_corr(df):
    print("Done")
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
