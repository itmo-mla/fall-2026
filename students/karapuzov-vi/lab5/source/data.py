import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "plots"

OPENML_NAME = "boston"
OPENML_VERSION = 1
PRICE_COLUMN = "MEDV"
TARGET_COLUMN = "high_value"
PRICE_THRESHOLD = 21.2
DROPPED_COLUMNS = ["B"]
FEATURE_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "LSTAT",
]
PLOT_NAMES = {
    "CRIM": "преступность на душу",
    "ZN": "крупные участки, %",
    "INDUS": "промзоны, %",
    "CHAS": "у реки Чарльз",
    "NOX": "оксиды азота",
    "RM": "комнат в доме",
    "AGE": "дома старше 1940, %",
    "DIS": "до центров занятости",
    "RAD": "доступ к шоссе",
    "TAX": "налог на имущество",
    "PTRATIO": "учеников на учителя",
    "LSTAT": "низкий статус, %",
    PRICE_COLUMN: "цена, тыс. $",
    TARGET_COLUMN: "дороже медианы",
}
CLASS_NAMES = {0: "не дороже медианы", 1: "дороже медианы"}
STRONG_FEATURES = ["LSTAT", "RM", "PTRATIO", "AGE"]
BOX_COLOR = "#4C72B0"
POSITIVE_COLOR = "#C44E52"


def load_raw_dataframe() -> pd.DataFrame:
    bunch = fetch_openml(name=OPENML_NAME, version=OPENML_VERSION, as_frame=True, parser="auto")
    frame = pd.DataFrame(bunch.data).copy()
    expected = FEATURE_COLUMNS + DROPPED_COLUMNS
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise RuntimeError(f"В OpenML boston нет колонок {missing}: {list(frame.columns)}")

    price = pd.Series(bunch.target, index=frame.index, name=PRICE_COLUMN).astype(float)
    if not np.isclose(float(price.median()), PRICE_THRESHOLD):
        raise RuntimeError(
            f"Медиана MEDV равна {float(price.median())}, ожидалась фиксированная {PRICE_THRESHOLD}."
        )

    kept = frame[FEATURE_COLUMNS].astype(float).copy()
    kept[PRICE_COLUMN] = price.to_numpy()
    kept[TARGET_COLUMN] = (kept[PRICE_COLUMN] >= PRICE_THRESHOLD).astype(int)
    return kept


def run_eda(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN]
    price = df[PRICE_COLUMN]

    print("\n=== Размер ===")
    print(f"районов: {len(df)}, признаков в матрице: {len(FEATURE_COLUMNS)}")
    print(f"пропусков: {int(df.isna().sum().sum())}, полных дубликатов: {int(df.duplicated().sum())}")
    print(f"столбец B в матрицу не входит")
    print(f"цена у потолка 50: {int((price == 50).sum())} районов")
    print(f"ZN равен нулю: {int((df['ZN'] == 0).sum())} ({(df['ZN'] == 0).mean():.1%}), это доля участков, не пропуск")

    print("\n=== Цель ===")
    print(f"MEDV, тыс. $: min {price.min():.1f}, медиана {price.median():.1f}, max {price.max():.1f}")
    print(f"класс 1: MEDV >= {PRICE_THRESHOLD}")
    counts = target.value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {CLASS_NAMES[int(label)]}: {int(count)} ({count / len(df):.1%})")

    print("\n=== Std признаков до масштаба (ddof=0) ===")
    spread = features.std(ddof=0).sort_values(ascending=False)
    print(spread.round(3).to_string())

    print("\n=== Медианы признаков по классам ===")
    by_class = df.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].median()
    by_class.index = [CLASS_NAMES[int(label)] for label in by_class.index]
    print(by_class.round(3).T.to_string())

    print("\n=== Корреляция Пирсона с классом ===")
    print(features.corrwith(target).sort_values().round(3).to_string())
    print("\n=== Корреляция Пирсона с ценой MEDV ===")
    print(features.corrwith(price).sort_values().round(3).to_string())

    feature_corr = features.corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    print("\n=== Максимальная |корреляция| между признаками ===")
    print(upper.stack().sort_values(ascending=False).head(5).round(3).to_string())

    print("\n=== Выбросы по правилу 1.5 IQR (не удаляем) ===")
    for column in FEATURE_COLUMNS + [PRICE_COLUMN]:
        series = df[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        print(f"  {column}: {int(mask.sum())} ({mask.mean():.1%})")

    _plot_price(price)
    _plot_class_balance(target)
    _plot_feature_scale(spread)
    _plot_correlation(df)
    _plot_strong_features(df)
    _plot_rooms_status(df)
    print(f"\nГрафики: {PLOTS_DIR}")


def _plot_price(price: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(price, bins=30, color=BOX_COLOR, label="число районов")
    ax.axvline(PRICE_THRESHOLD, color=POSITIVE_COLOR, linewidth=1.6, label=f"медиана {PRICE_THRESHOLD}")
    ax.set_title("Медианная цена жилья и порог класса")
    ax.set_xlabel("MEDV, тысячи долларов")
    ax.set_ylabel("Число районов")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_price_threshold.png", dpi=120)
    plt.close(fig)


def _plot_class_balance(target: pd.Series) -> None:
    counts = target.value_counts().sort_index()
    labels = [CLASS_NAMES[int(label)] for label in counts.index]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(labels, counts.to_numpy(), color=[BOX_COLOR, POSITIVE_COLOR])
    ax.set_title(f"Класс: цена района не ниже медианы {PRICE_THRESHOLD}")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число районов")
    ax.legend(bars, labels, title="Класс", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_class_balance.png", dpi=120)
    plt.close(fig)


def _plot_feature_scale(spread: pd.Series) -> None:
    ordered = spread.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [PLOT_NAMES[column] for column in ordered.index]
    ax.barh(labels, ordered.to_numpy(), color=BOX_COLOR, label="std по всей выборке")
    ax.set_title("Разброс признаков до стандартизации")
    ax.set_xlabel("Стандартное отклонение в исходных единицах")
    ax.set_ylabel("Признак")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_feature_scale.png", dpi=120)
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame) -> None:
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    renamed = df[columns].rename(columns=PLOT_NAMES)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        renamed.corr(),
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "корреляция Пирсона"},
    )
    ax.set_title("Корреляции признаков и класса «дороже медианы»")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_correlation.png", dpi=120)
    plt.close(fig)


def _plot_strong_features(df: pd.DataFrame) -> None:
    plot_df = df[STRONG_FEATURES + [TARGET_COLUMN]].copy()
    plot_df["класс"] = plot_df[TARGET_COLUMN].map(CLASS_NAMES)
    class_order = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, column in zip(axes.ravel(), STRONG_FEATURES):
        sns.boxplot(
            data=plot_df,
            x="класс",
            y=column,
            hue="класс",
            order=class_order,
            hue_order=class_order,
            palette=[BOX_COLOR, POSITIVE_COLOR],
            fliersize=2,
            legend=False,
            ax=ax,
        )
        ax.set_title(PLOT_NAMES[column])
        ax.set_xlabel("Класс")
        ax.set_ylabel(PLOT_NAMES[column])
    fig.suptitle("Признаки с самой сильной связью с классом цены")
    fig.legend(
        handles=[
            Patch(facecolor=BOX_COLOR, label=CLASS_NAMES[0]),
            Patch(facecolor=POSITIVE_COLOR, label=CLASS_NAMES[1]),
        ],
        title="Класс",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(PLOTS_DIR / "05_strong_features.png", dpi=120)
    plt.close(fig)


def _plot_rooms_status(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["класс"] = plot_df[TARGET_COLUMN].map(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="RM",
        y="LSTAT",
        hue="класс",
        hue_order=[CLASS_NAMES[0], CLASS_NAMES[1]],
        palette=[BOX_COLOR, POSITIVE_COLOR],
        alpha=0.8,
        s=28,
        ax=ax,
    )
    ax.set_title("Комнаты и доля жителей с низким статусом")
    ax.set_xlabel(PLOT_NAMES["RM"])
    ax.set_ylabel(PLOT_NAMES["LSTAT"])
    ax.legend(title="Класс")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_rooms_lstat.png", dpi=120)
    plt.close(fig)


def prepare_data(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if df is None:
        df = load_raw_dataframe()

    y = df[TARGET_COLUMN].astype(int)
    y.name = TARGET_COLUMN
    X = df[FEATURE_COLUMNS].astype(float)
    if PRICE_COLUMN in X.columns or "B" in X.columns:
        raise RuntimeError("В матрицу попала цена или столбец B.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS)
    X_train_design = _add_intercept(X_train_scaled)
    X_test_design = _add_intercept(X_test_scaled)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train.name = TARGET_COLUMN
    y_test.name = TARGET_COLUMN

    features_only = X_train_design[FEATURE_COLUMNS]
    if not np.allclose(features_only.mean().to_numpy(), 0.0, atol=1e-8):
        raise RuntimeError("После StandardScaler среднее train не ноль: scaler обучен не на train.")
    if not np.allclose(features_only.std(ddof=0).to_numpy(), 1.0, atol=1e-6):
        raise RuntimeError("После StandardScaler std train (ddof=0) не единица.")
    if not np.array_equal(X_train_design["intercept"].to_numpy(), np.ones(len(X_train_design))):
        raise RuntimeError("Столбец intercept должен быть единичным и не проходить через scaler.")
    if set(np.unique(y_train)) - {0, 1} or set(np.unique(y_test)) - {0, 1}:
        raise RuntimeError("Метки логистической регрессии должны быть 0 и 1.")

    return {
        "X_train": X_train_design,
        "X_test": X_test_design,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_train_design.columns),
        "scaler": scaler,
        "price_threshold": PRICE_THRESHOLD,
    }


def _add_intercept(features: pd.DataFrame) -> pd.DataFrame:
    design = features.copy()
    design.insert(0, "intercept", 1.0)
    return design


def as_numpy(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X уже содержит свободный член. y — это 0/1, не метки отступа −1/+1."""
    return (
        data["X_train"].to_numpy(dtype=float),
        data["X_test"].to_numpy(dtype=float),
        data["y_train"].to_numpy(dtype=float),
        data["y_test"].to_numpy(dtype=float),
    )


def print_prepared_summary(data: dict) -> None:
    X_train = data["X_train"]
    X_test = data["X_test"]
    raw_train = X_train.drop(columns="intercept").to_numpy() * data["scaler"].scale_ + data["scaler"].mean_
    raw_design = np.column_stack([np.ones(len(raw_train)), raw_train])
    scaled_design = X_train.to_numpy(dtype=float)
    raw_singular = np.linalg.svd(raw_design, compute_uv=False)
    scaled_singular = np.linalg.svd(scaled_design, compute_uv=False)

    print("\n=== После подготовки ===")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print("порог MEDV:", data["price_threshold"])
    print("столбцы:", data["feature_names"])
    print("пропуски train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
    print("max |среднее| признаков train:", float(np.max(np.abs(X_train.drop(columns="intercept").mean()))))
    feature_std = X_train.drop(columns="intercept").std(ddof=0)
    print("std признаков train ddof=0, min/max:", float(feature_std.min()), float(feature_std.max()))
    print(
        "доля класса 1 train/test:",
        round(float(data["y_train"].mean()), 3),
        round(float(data["y_test"].mean()), 3),
    )
    print("метки train:", data["y_train"].value_counts().sort_index().to_dict())
    print("метки test:", data["y_test"].value_counts().sort_index().to_dict())
    print(
        "число обусловленности матрицы со свободным членом:",
        round(float(raw_singular.max() / raw_singular.min()), 1),
        "до масштаба,",
        round(float(scaled_singular.max() / scaled_singular.min()), 3),
        "после",
    )


if __name__ == "__main__":
    raw = load_raw_dataframe()
    run_eda(raw)
    prepared = prepare_data(raw)
    print_prepared_summary(prepared)
