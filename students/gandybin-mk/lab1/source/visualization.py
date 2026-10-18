import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_correlations(X, y, feature_names):
    """Возвращает корреляции обучающих признаков с целевой переменной."""
    correlations = [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
    return pd.Series(correlations, index=feature_names, name="correlation")


def analyze_margins(margins):
    """Возвращает характеристики отступов и число объектов по их знаку."""
    margins = np.asarray(margins)
    return {
        "mean_margin": float(margins.mean()),
        "min_margin": float(margins.min()),
        "max_margin": float(margins.max()),
        "negative_count": int(np.count_nonzero(margins < 0)),
        "positive_count": int(np.count_nonzero(margins > 0)),
        "zero_count": int(np.count_nonzero(margins == 0)),
    }


def plot_margins(margins, title):
    """Строит гистограмму отступов с границей нуля и возвращает Figure."""
    figure, ax = plt.subplots(figsize=(10, 5))
    ax.hist(margins, bins=30, alpha=0.8)
    ax.axvline(0, linestyle="--", linewidth=2, label="M = 0")
    ax.set(xlabel="Отступ", ylabel="Количество объектов", title=title)
    ax.legend()
    ax.grid()
    return figure


def plot_quality_history(histories, title, xlabel="Шаг SGD"):
    """Строит истории Q из словаря {название: значения} и возвращает Figure."""
    figure, ax = plt.subplots(figsize=(11, 5))
    for name, history in histories.items():
        ax.plot(history, label=name)
    ax.set(xlabel=xlabel, ylabel="Рекуррентная оценка loss", title=title)
    ax.legend()
    ax.grid()
    return figure


def plot_confusion_matrix(matrix, title):
    """Отображает готовую матрицу ошибок для классов [-1, +1]."""
    figure, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=[-1, 1])
    ax.set_yticks([0, 1], labels=[-1, 1])
    ax.set(xlabel="Предсказанный класс", ylabel="Истинный класс", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="white" if matrix[i, j] < matrix.max() / 2 else "black",
            )
    return figure
