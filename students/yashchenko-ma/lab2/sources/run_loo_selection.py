"""
Запускной скрипт для пунктов 3-4 задания: подбор k методом LOO,
построение графиков эмпирического риска. Запускать из папки sources/:

    python run_loo_selection.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_and_split
from loo import loo_risk_curve, best_k

IMAGES_DIR = os.path.join("..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

'''
LOO - для каждого объекта $x_i$ обучаем модель на всех остальных X из выборки: из которой исключаем проверяемый xi ($X^\ell\setminus\{x_i\}$) 
и проверяем, правильно ли она классифицирует именно этот, "отложенный", объект.
'''
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"train: {X_train.shape}, баланс классов {np.bincount(y_train)}")

    k_values = np.arange(1, 41)
    k_risk, risks, f1s = best_k(X_train, y_train, k_values, criterion="risk")
    k_f1, _, _ = best_k(X_train, y_train, k_values, criterion="f1")
    print(f"k* по 0/1-риску = {k_risk} (риск={risks[k_risk-1]:.4f}, F1={f1s[k_risk-1]:.4f})")
    print(f"k* по F1        = {k_f1} (риск={risks[k_f1-1]:.4f}, F1={f1s[k_f1-1]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_values, risks, marker="o", markersize=3, label="LOO-риск (0/1)")
    axes[0].axvline(k_risk, color="red", linestyle="--", label=f"k*={k_risk}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("LOO-риск (доля ошибок)")
    axes[0].set_title("Эмпирический риск (0/1) в зависимости от k")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_values, f1s, marker="o", markersize=3, color="darkorange", label="LOO F1 (класс 1)")
    axes[1].axvline(k_f1, color="red", linestyle="--", label=f"k*={k_f1}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("F1 (класс 'хорошее вино')")
    axes[1].set_title("F1-мера в зависимости от k")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "loo_risk_curve.png")
    plt.savefig(out_path, dpi=130)
    print(f"сохранено: {out_path}")

    return k_risk


if __name__ == "__main__":
    main()
