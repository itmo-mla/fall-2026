"""
Запускной скрипт для пунктов 6-8 задания: отбор эталонов, визуализация,
сравнение качества KNN с отбором эталонов и без. Запускать из папки
sources/:

    python run_prototype_selection.py

Классификатор для оценки качества -- собственная реализация KNNVote(k=1)
(жёсткое голосование, ровно классификатор 1NN, для которого определён
алгоритм отбора эталонов), а не sklearn -- согласно правилу "свои
алгоритмы без готовых реализаций".
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA          # только визуализация, не алгоритм
from sklearn.metrics import accuracy_score, f1_score

from data import load_and_split
from prototype_selection import greedy_prototype_selection
from knn_parzen import KNNVote

IMAGES_DIR = os.path.join("..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    # --- пункт 6: отбор эталонов ---
    omega, history = greedy_prototype_selection(X_train, y_train, verbose=True)
    print(f"\nЭталонов отобрано: {len(omega)} из {len(y_train)} "
          f"(сжатие {100*(1-len(omega)/len(y_train)):.1f}%)")
    print(f"Ошибок 1NN(Omega) в начале: {history[0]} ({history[0]/len(y_train):.4f}), "
          f"в конце: {history[-1]} ({history[-1]/len(y_train):.4f})")
    print("Баланс классов в эталонах:", np.bincount(y_train[omega]))
    print("Баланс классов в исходной выборке:", np.bincount(y_train))

    # --- пункт 7: визуализация ---
    rest = np.setdiff1d(np.arange(len(y_train)), omega)
    pca = PCA(n_components=2, random_state=42).fit(X_train)
    Z = pca.transform(X_train)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for cls, marker in [(0, "o"), (1, "^")]:
        ax.scatter(Z[rest][y_train[rest] == cls, 0], Z[rest][y_train[rest] == cls, 1],
                   c="tab:blue" if cls == 0 else "tab:orange", alpha=0.2, marker=marker, s=18,
                   label=f"обычный объект, класс {cls}")
        ax.scatter(Z[omega][y_train[omega] == cls, 0], Z[omega][y_train[omega] == cls, 1],
                   c="darkred" if cls == 0 else "black", marker=marker, s=110, edgecolors="white",
                   linewidths=1.0, label=f"эталон, класс {cls}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Отобранные эталоны (жадный CCV-критерий)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=3,
             color="tab:green", label="ошибки 1NN(Omega) на всей выборке")
    ax2.set_xlabel("число эталонов в Omega")
    ax2.set_ylabel("число ошибок 1NN(Omega) на всей выборке")
    ax2.set_title("Сходимость критерия CCV(Omega) (~LOO при k=1)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "prototype_selection.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nсохранено: {out_path}")

    # --- пункт 8: сравнение качества с отбором эталонов и без ---
    print("\n--- сравнение качества (собственная реализация KNNVote, k=1) ---")
    results = {}
    for name, Xr, yr in [
        ("Полная выборка, 1NN", X_train, y_train),
        (f"Эталоны Omega, 1NN ({len(omega)} из {len(y_train)})", X_train[omega], y_train[omega]),
    ]:
        model = KNNVote(k=1).fit(Xr, yr)
        t0 = time.perf_counter()
        pred = model.predict(X_test)
        dt = time.perf_counter() - t0
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, pos_label=1)
        results[name] = (acc, f1, dt, len(yr))
        print(f"{name}: accuracy={acc:.4f} F1={f1:.4f} predict_time={dt*1000:.2f}ms n_ref={len(yr)}")

    full_key = "Полная выборка, 1NN"
    omega_key = [k for k in results if k.startswith("Эталоны")][0]
    print(f"\nСжатие выборки: {100*(1 - results[omega_key][3]/results[full_key][3]):.1f}%")
    print(f"Ускорение predict: {results[full_key][2] / results[omega_key][2]:.2f}x")


if __name__ == "__main__":
    main()
