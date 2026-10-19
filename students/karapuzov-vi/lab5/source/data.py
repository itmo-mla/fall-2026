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

# Pima Indians Diabetes, OpenML data id 37. Не путать со sklearn.load_diabetes:
# тот датасет — регрессия с непрерывной целью, здесь цель бинарная.
OPENML_DATA_ID = 37

TARGET_COLUMN = "diabetes"
FEATURE_COLUMNS = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
# Ноль здесь не измерение, а пропуск в исходной таблице. У беременностей ноль бывает.
ZERO_AS_MISSING = ["plas", "pres", "skin", "insu", "mass"]

PLOT_NAMES = {
    "preg": "беременности, число",
    "plas": "глюкоза, мг/дл",
    "pres": "давление, мм рт. ст.",
    "skin": "кожная складка, мм",
    "insu": "инсулин, мкЕд/мл",
    "mass": "ИМТ, кг/м²",
    "pedi": "наследственность",
    "age": "возраст, лет",
    TARGET_COLUMN: "диабет",
}
CLASS_NAMES = {0: "нет диабета", 1: "диабет"}
STRONG_FEATURES = ["plas", "mass", "age", "insu"]
BOX_COLOR = "#4C72B0"
POSITIVE_COLOR = "#C44E52"


def load_raw_dataframe() -> pd.DataFrame:
    """Таблица с нулями там, где в источнике не было измерения.

    Цель уже 0/1: 1 — tested_positive. CSV в репозиторий не кладётся,
    sklearn скачивает OpenML в свой кэш.
    """
    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    frame = bunch.data.copy()
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"В OpenML id {OPENML_DATA_ID} нет колонок {missing}: {list(frame.columns)}")

    labels = bunch.target.astype(str)
    expected = {"tested_negative", "tested_positive"}
    if set(labels.unique()) - expected:
        raise RuntimeError(f"Неизвестные метки: {sorted(labels.unique())}")

    frame = frame[FEATURE_COLUMNS].copy()
    frame[TARGET_COLUMN] = (labels.to_numpy() == "tested_positive").astype(int)
    return frame


def _with_missing_zeros(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in ZERO_AS_MISSING:
        cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def run_eda(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    cleaned = _with_missing_zeros(df)
    features = cleaned[FEATURE_COLUMNS]
    target = cleaned[TARGET_COLUMN]

    print("\n=== Размер и пропуски ===")
    print(f"объектов: {len(df)}, признаков: {len(FEATURE_COLUMNS)}")
    print(f"пропусков в файле OpenML: {int(df.isna().sum().sum())}")
    print(f"полных дубликатов строк: {int(df.duplicated().sum())}")
    print("нули, которые считаем пропуском:")
    for column in ZERO_AS_MISSING:
        count = int((df[column] == 0).sum())
        print(f"  {column}: {count} ({count / len(df):.1%})")
    print(f"беременности с нулём (это измерение, не пропуск): {int((df['preg'] == 0).sum())}")

    print("\n=== Цель ===")
    counts = target.value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {CLASS_NAMES[int(label)]}: {int(count)} ({count / len(df):.1%})")

    print("\n=== Доля диабета, если измерение пропущено ===")
    for column in ZERO_AS_MISSING:
        missing = features[column].isna()
        print(
            f"  {column}: пропуск {target[missing].mean():.3f}, "
            f"измерено {target[~missing].mean():.3f}, n_пропусков={int(missing.sum())}"
        )

    print("\n=== Std наблюдаемых значений до масштаба (ddof=0, нули-пропуски убраны) ===")
    spread = features.std(ddof=0).sort_values(ascending=False)
    print(spread.round(3).to_string())

    print("\n=== Медианы по классам, нули-пропуски не входят ===")
    by_class = cleaned.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].median()
    by_class.index = [CLASS_NAMES[int(label)] for label in by_class.index]
    print(by_class.round(2).T.to_string())

    print("\n=== Корреляция Пирсона с диабетом, попарно по наблюдаемым ===")
    pearson = features.corrwith(target).sort_values()
    print(pearson.round(3).to_string())

    feature_corr = features.corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    print("\n=== Максимальная |корреляция| между признаками ===")
    print(upper.stack().sort_values(ascending=False).head(5).round(3).to_string())

    negative_glucose = cleaned.loc[target == 0, "plas"].median()
    positive = cleaned.loc[target == 1, "plas"].dropna()
    below = float((positive < negative_glucose).mean())
    print("\n=== Перекрытие по глюкозе ===")
    print(f"медиана глюкозы без диабета: {negative_glucose:.1f}")
    print(f"доля больных с глюкозой ниже этой медианы: {below:.1%}")

    print("\n=== Выбросы по правилу 1.5 IQR среди наблюдаемых (не удаляем) ===")
    for column in FEATURE_COLUMNS:
        observed = features[column].dropna()
        q1 = observed.quantile(0.25)
        q3 = observed.quantile(0.75)
        iqr = q3 - q1
        mask = (features[column] < q1 - 1.5 * iqr) | (features[column] > q3 + 1.5 * iqr)
        print(f"  {column}: {int(mask.sum())} ({mask.mean():.1%})")

    _plot_class_balance(target)
    _plot_missing(df)
    _plot_feature_scale(spread)
    _plot_correlation(cleaned)
    _plot_strong_features(cleaned)
    _plot_glucose_bmi(cleaned)
    print(f"\nГрафики: {PLOTS_DIR}")


def _plot_class_balance(target: pd.Series) -> None:
    counts = target.value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "класс": [CLASS_NAMES[int(label)] for label in counts.index],
            "число женщин": counts.to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(
        plot_df["класс"],
        plot_df["число женщин"],
        color=[BOX_COLOR, POSITIVE_COLOR],
    )
    ax.set_title("Баланс классов: диабет у женщин Pima")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число женщин")
    ax.legend(bars, plot_df["класс"].tolist(), title="Класс")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)


def _plot_missing(df: pd.DataFrame) -> None:
    rates = pd.Series({column: (df[column] == 0).mean() for column in ZERO_AS_MISSING})
    rates = rates.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [PLOT_NAMES[column] for column in rates.index]
    ax.barh(labels, rates.to_numpy(), color=BOX_COLOR, label="доля нулей в исходной таблице")
    ax.set_xlim(0, 1)
    ax.set_title("Нули, которые в этой таблице означают пропуск измерения")
    ax.set_xlabel("Доля строк")
    ax.set_ylabel("Признак")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_missing_zeros.png", dpi=120)
    plt.close(fig)


def _plot_feature_scale(spread: pd.Series) -> None:
    ordered = spread.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [PLOT_NAMES[column] for column in ordered.index]
    ax.barh(labels, ordered.to_numpy(), color=BOX_COLOR, label="std наблюдаемых значений")
    ax.set_title("Разброс признаков до стандартизации")
    ax.set_xlabel("Стандартное отклонение в исходных единицах")
    ax.set_ylabel("Признак")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_feature_scale.png", dpi=120)
    plt.close(fig)


def _plot_correlation(cleaned: pd.DataFrame) -> None:
    renamed = cleaned.rename(columns=PLOT_NAMES)
    order = [PLOT_NAMES[column] for column in FEATURE_COLUMNS + [TARGET_COLUMN]]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        renamed[order].corr(),
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "корреляция Пирсона"},
    )
    ax.set_title("Корреляции признаков и диабета, пропуски не заполнены")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_correlation.png", dpi=120)
    plt.close(fig)


def _plot_strong_features(cleaned: pd.DataFrame) -> None:
    plot_df = cleaned[STRONG_FEATURES + [TARGET_COLUMN]].copy()
    plot_df["класс"] = plot_df[TARGET_COLUMN].map(CLASS_NAMES)
    class_order = ["нет диабета", "диабет"]
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
    fig.suptitle("Признаки с самой заметной связью с диабетом")
    fig.legend(
        handles=[
            Patch(facecolor=BOX_COLOR, label="нет диабета"),
            Patch(facecolor=POSITIVE_COLOR, label="диабет"),
        ],
        title="Класс",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(PLOTS_DIR / "05_strong_features.png", dpi=120)
    plt.close(fig)


def _plot_glucose_bmi(cleaned: pd.DataFrame) -> None:
    plot_df = cleaned.dropna(subset=["plas", "mass"]).copy()
    plot_df["класс"] = plot_df[TARGET_COLUMN].map(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="plas",
        y="mass",
        hue="класс",
        hue_order=["нет диабета", "диабет"],
        palette=[BOX_COLOR, POSITIVE_COLOR],
        alpha=0.75,
        s=28,
        ax=ax,
    )
    ax.set_title("Глюкоза и ИМТ: облака классов пересекаются")
    ax.set_xlabel(PLOT_NAMES["plas"])
    ax.set_ylabel(PLOT_NAMES["mass"])
    ax.legend(title="Класс")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_glucose_bmi.png", dpi=120)
    plt.close(fig)


def prepare_data(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Матрица для логистической регрессии.

    Ноль в глюкозе, давлении, складке, инсулине и ИМТ заменяется на NaN.
    Медиана для заполнения считается только на train. StandardScaler тоже
    учится только на train и не трогает столбец единиц: свободный член
    добавляется после масштаба. Метки остаются 0 и 1.
    """
    if df is None:
        df = load_raw_dataframe()

    cleaned = _with_missing_zeros(df)
    y = cleaned[TARGET_COLUMN].astype(int)
    y.name = TARGET_COLUMN
    X = cleaned[FEATURE_COLUMNS].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
    )

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
        "medians": medians,
        "zero_as_missing": list(ZERO_AS_MISSING),
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
    features = X_train.drop(columns="intercept").to_numpy(dtype=float)
    raw_train = features * data["scaler"].scale_ + data["scaler"].mean_
    raw_design = np.column_stack([np.ones(len(raw_train)), raw_train])
    scaled_design = X_train.to_numpy(dtype=float)

    print("\n=== После подготовки ===")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print("столбцы:", data["feature_names"])
    print("медианы train, которыми заполнены пропуски:")
    print(data["medians"].round(3).to_string())
    print("пропуски train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
    print("max |среднее| признаков train:", float(np.max(np.abs(X_train.drop(columns="intercept").mean()))))
    feature_std = X_train.drop(columns="intercept").std(ddof=0)
    print("std признаков train ddof=0, min/max:", float(feature_std.min()), float(feature_std.max()))
    print(
        "доля диабета train/test:",
        round(float(data["y_train"].mean()), 3),
        round(float(data["y_test"].mean()), 3),
    )
    print("метки train:", data["y_train"].value_counts().sort_index().to_dict())
    print("метки test:", data["y_test"].value_counts().sort_index().to_dict())
    print("константа «всегда нет диабета», доля верных на test:", round(float(1 - data["y_test"].mean()), 3))

    raw_singular = np.linalg.svd(raw_design, compute_uv=False)
    scaled_singular = np.linalg.svd(scaled_design, compute_uv=False)
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
