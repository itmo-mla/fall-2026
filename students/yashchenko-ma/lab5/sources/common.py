"""Общие данные и стиль графиков для ЛР4.

Задача регрессии: предсказать непрерывный torque по остальным признакам + Type.
Данные берутся через load_raw() из data.py (data.py не изменён).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_raw, FEATURES, RANDOM_STATE

TARGET = "torque"
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

# Палитра: синий / оранжевый / аквамарин (светлая тема), поверхность #fcfcfb
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 7, "font.size": 11,
        "legend.frameon": False,
    })


def save_fig(fig, name):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.normpath(os.path.join(IMAGE_DIR, name))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] сохранено: {path}")


def get_regression_data(test_size=0.25, random_state=RANDOM_STATE):
    """Полная выборка (10000 строк), цель - torque.
    Признаки: air_temp, process_temp, rot_speed, tool_wear + type_L, type_M.
    StandardScaler обучается только на train."""
    df = load_raw()
    feats = [f for f in FEATURES if f != TARGET]
    dummies = pd.get_dummies(df["Type"], prefix="type", drop_first=True).astype(float)
    X = pd.concat([df[feats], dummies], axis=1)
    names = list(X.columns)
    y = df[TARGET].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, names
