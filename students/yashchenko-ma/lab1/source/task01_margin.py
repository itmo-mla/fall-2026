"""
Задача 1: реализовать вычисление отступа объекта (визуализировать, проанализировать).

Это root file - здесь же живёт загрузка/предобработка данных,
которую переиспользуют все последующие task-файлы (task02...task10) через
прямой импорт друг друга по цепочке.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

from download_data import download_ai4i_dataset

NUMERIC_COLS = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def get_dataset_path(dest_csv="ai4i2020.csv"):
    """Пробует скачать датасет напрямую с Kaggle; при неудаче (нет ключей
    в заготовке download_data.py или нет сети) использует локальную копию."""
    try:
        return download_ai4i_dataset(dest_csv)
    except Exception as e:
        print("Не удалось скачать напрямую с Kaggle:")
        print(f"  {e}")
        print(f"Используется уже имеющаяся локальная копия {dest_csv}.")
        return dest_csv


def load_and_prepare(path=None, test_size=0.2, random_state=42):
    """Загрузка + предобработка: one-hot для Type, стандартизация числовых
    признаков, метки в {-1,+1}, train/test split."""
    if path is None:
        path = get_dataset_path()
    df = pd.read_csv(path)

    type_dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=float)
    X_num = df[NUMERIC_COLS].astype(float).copy()
    X = pd.concat([X_num, type_dummies], axis=1)
    feature_names = list(X.columns)

    y = df["Machine failure"].values.astype(float)
    y = np.where(y == 1, 1.0, -1.0)

    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    n_num = len(NUMERIC_COLS)
    mu = X_train[:, :n_num].mean(axis=0)
    sigma = X_train[:, :n_num].std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train[:, :n_num] = (X_train[:, :n_num] - mu) / sigma
    X_test[:, :n_num] = (X_test[:, :n_num] - mu) / sigma

    return X_train, X_test, y_train, y_test, feature_names, (mu, sigma)


def add_bias(X):
    """Добавляет столбец единиц для свободного члена w0."""
    return np.hstack([np.ones((X.shape[0], 1)), X]) # склеивает массивы по горизонтали (по столбцам), кладя их бок о бок


def margin(w, X, y):
    """Отступ объекта: M_i = y_i * <w, x_i>. X уже с bias-столбцом."""
    return y * (X @ w)


def correlation_init(X, y):
    """Инициализация весов через корреляцию признака с целью (нужна в задаче 8.1,
    но заодно используется здесь как демонстрационные веса для отступа)."""
    n_features = X.shape[1]
    w = np.zeros(n_features)
    for j in range(n_features):
        xj = X[:, j]
        if xj.std() > 0:
            w[j] = np.corrcoef(xj, y)[0, 1] # коэффициент корреляции Пирсона между двумя (или больше) наборами чисел
    return w


def plot_margin_distribution(w, X, y, title, filename):
    """Строит и сохраняет отсортированный график отступа (переиспользуется
    и в task01, и в task04 - для сравнения 'до' и 'после' обучения)."""
    M = margin(w, X, y)
    M_sorted = np.sort(M)
    colors = np.where(M_sorted < 0, "red", "steelblue")
 
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(M_sorted)), M_sorted, color=colors, width=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Объекты (отсортированы по отступу)")
    ax.set_ylabel("Отступ M(x)")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(facecolor="red", label="M < 0 (ошибка классификации)"),
        Patch(facecolor="steelblue", label="M >= 0 (верно классифицирован)"),
    ])
    plt.tight_layout()
    os.makedirs("../images", exist_ok=True)
    plt.savefig(f"../images/{filename}", dpi=120)
    plt.close(fig)
    print(f"График сохранён: images/{filename} (доля M<0: {(M < 0).mean()*100:.2f}%)")
    return M

if __name__ == "__main__":
    csv_path = get_dataset_path()
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare(path=csv_path)
    Xb_train = add_bias(X_train)
 
    # --- Корреляционная матрица исходных числовых признаков и цели ---
    # (общий контекст для анализа отступа: слабая попарная корреляция признаков
    # с целью объясняет, почему линейная модель не может дать высокий recall
    # без дополнительных приёмов — см. задачи 7-8).
    df = pd.read_csv(csv_path)
    corr_cols = NUMERIC_COLS + ["Machine failure"]
    corr = df[corr_cols].corr()
 
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Коэффициент корреляции Пирсона")
    ax.set_title("Задача 1: корреляционная матрица числовых признаков и цели")
    plt.tight_layout()
    os.makedirs("../images", exist_ok=True)
    plt.savefig("../images/dataset_correlation_matrix.png", dpi=120)
    plt.close(fig)
    print("График сохранён: images/task01_dataset_correlation_matrix.png")
 
    # --- Отступ при случайной инициализации весов (для сравнения) ---
    rng = np.random.RandomState(42)
    w_random = rng.normal(scale=0.01, size=Xb_train.shape[1])
    plot_margin_distribution(w_random, Xb_train, y_train,
                              "Задача 1: отступы при случайной инициализации весов",
                              "task01_random_weights_init.png")
 
    # --- Отступ при корреляционной инициализации весов (основной вариант задачи 1) ---
    w0 = correlation_init(Xb_train, y_train)
    M = plot_margin_distribution(w0, Xb_train, y_train,
                                  "Задача 1: отступы объектов при корреляционной инициализации весов",
                                  "task01_correlation_weights_init.png")
 
    print("\nВеса (корреляционная инициализация):")
    for name, val in zip(["bias"] + feature_names, w0):
        print(f"  {name:25s} {val:+.4f}")
    print(f"\nВсего объектов: {len(M)}")
    print(f"Отрицательный отступ (ошибки): {(M < 0).sum()} ({(M < 0).mean()*100:.2f}%)")
    print(f"M статистика: min={M.min():.3f}, median={np.median(M):.3f}, max={M.max():.3f}")
 
    print("\nАнализ: при корреляционной инициализации классификатор ещё не "
          "обучен - отступ симметричен около нуля (~50% M<0), т.к. bias не "
          "смещён и веса не учитывают дисбаланс классов (96.6%/3.4%). При "
          "случайной инициализации малыми весами картина ещё хуже (доля M<0 "
          "выше) - веса вообще не согласованы с данными.")