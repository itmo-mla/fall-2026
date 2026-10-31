from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "plots"

OPENML_NAME = "banknote-authentication"
OPENML_VERSION = 1

COLUMN_NAMES = {
    "V1": "variance",
    "V2": "skewness",
    "V3": "curtosis",
    "V4": "entropy",
}
FEATURE_COLUMNS = list(COLUMN_NAMES.values())
PLOT_NAMES = {
    "variance": "дисперсия вейвлета",
    "skewness": "асимметрия",
    "curtosis": "эксцесс",
    "entropy": "энтропия",
}
GENUINE_RAW = "1"
FORGED_RAW = "2"
GENUINE = -1
FORGED = 1
CLASS_NAMES = {
    GENUINE: "подлинная",
    FORGED: "подделка",
}

VISUAL_FEATURES = ["variance", "skewness"]

SOLVER_TRAIN_SIZE = 400

def load_raw_dataframe() -> pd.DataFrame:
    bunch = fetch_openml(
        name=OPENML_NAME,
        version=OPENML_VERSION,
        as_frame=True,
        parser="auto",
    )
    frame = bunch.data.rename(columns=COLUMN_NAMES).copy()
    frame["raw_class"] = bunch.target.astype(str).to_numpy()
    return frame


def encode_labels(raw_class: pd.Series) -> pd.Series:
    counts = raw_class.value_counts()
    if counts.get(GENUINE_RAW) != 762 or counts.get(FORGED_RAW) != 610:
        raise RuntimeError(
            "Коды классов OpenML не совпали с UCI (762 подлинных, 610 подделок). "
            f"Сейчас: {counts.to_dict()}. Подпись подлинная/подделка надо проверить заново."
        )
    mapping = {GENUINE_RAW: GENUINE, FORGED_RAW: FORGED}
    labels = raw_class.map(mapping)
    if labels.isna().any():
        unknown = sorted(raw_class[labels.isna()].unique())
        raise RuntimeError(f"Неизвестные метки: {unknown}")
    return labels.astype(int)


def run_eda(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    labels = encode_labels(df["raw_class"])
    eda = df[FEATURE_COLUMNS].copy()
    eda["класс"] = labels.map(CLASS_NAMES)

    print("\n=== Размер и пропуски ===")
    print(f"объектов: {len(eda)}, признаков: {len(FEATURE_COLUMNS)}")
    print(eda[FEATURE_COLUMNS].isna().sum().to_string())
    print("\n=== Баланс ===")
    balance = eda["класс"].value_counts()
    print(balance.to_string())
    print((balance / len(eda)).round(3).to_string())

    print("\n=== Статистика ===")
    print(eda[FEATURE_COLUMNS].describe().round(3).to_string())
    print("\n=== Средние по классам ===")
    print(eda.groupby("класс")[FEATURE_COLUMNS].mean().round(3).to_string())
    print("\n=== Std признаков (до масштаба) ===")
    print(eda[FEATURE_COLUMNS].std(ddof=0).round(3).sort_values(ascending=False).to_string())

    print("\n=== Корреляция с подделкой (+1) ===")
    target = labels.rename("y")
    corr_with_target = eda[FEATURE_COLUMNS].corrwith(target).sort_values()
    print(corr_with_target.round(3).to_string())
    print("\n=== Корреляция признаков ===")
    print(eda[FEATURE_COLUMNS].corr().round(3).to_string())

    print("\n=== Выбросы по правилу 1.5 IQR (не удаляем) ===")
    for column in FEATURE_COLUMNS:
        q1 = eda[column].quantile(0.25)
        q3 = eda[column].quantile(0.75)
        iqr = q3 - q1
        mask = (eda[column] < q1 - 1.5 * iqr) | (eda[column] > q3 + 1.5 * iqr)
        print(f"{column}: {int(mask.sum())} ({mask.mean():.1%})")

    _plot_class_balance(eda)
    _plot_main_scatter(eda)
    _plot_boxplots(eda)
    _plot_correlation(eda[FEATURE_COLUMNS], target)
    _plot_feature_scale(eda)
    _plot_pairplot(eda)
    print(f"\nГрафики: {PLOTS_DIR}")


def _plot_class_balance(eda: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = [CLASS_NAMES[GENUINE], CLASS_NAMES[FORGED]]
    counts = eda["класс"].value_counts().reindex(order)
    plot_df = pd.DataFrame({"класс": order, "число": counts.to_numpy()})
    sns.barplot(data=plot_df, x="класс", y="число", hue="класс", legend="brief", ax=ax)
    ax.set_title("Баланс классов: подлинные и поддельные купюры")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число купюр")
    ax.get_legend().set_title("Класс")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)


def _plot_main_scatter(eda: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=eda,
        x="variance",
        y="skewness",
        hue="класс",
        hue_order=[CLASS_NAMES[GENUINE], CLASS_NAMES[FORGED]],
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Купюры в осях дисперсии и асимметрии вейвлета")
    ax.set_xlabel(PLOT_NAMES["variance"])
    ax.set_ylabel(PLOT_NAMES["skewness"])
    ax.legend(title="Класс")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_variance_skewness.png", dpi=120)
    plt.close(fig)


def _plot_boxplots(eda: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.boxplot(
            data=eda,
            x="класс",
            y=column,
            hue="класс",
            order=[CLASS_NAMES[GENUINE], CLASS_NAMES[FORGED]],
            hue_order=[CLASS_NAMES[GENUINE], CLASS_NAMES[FORGED]],
            legend="brief",
            ax=ax,
        )
        ax.set_title(PLOT_NAMES[column])
        ax.set_xlabel("Класс")
        ax.set_ylabel(PLOT_NAMES[column])
    legend = axes[0, 0].get_legend()
    handles = list(legend.legend_handles)
    legend_labels = [text.get_text() for text in legend.texts]
    for ax in axes.ravel():
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend(handles, legend_labels, title="Класс", loc="upper right")
    fig.suptitle("Признаки по классам, исходный масштаб")
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.savefig(PLOTS_DIR / "03_feature_boxplots.png", dpi=120)
    plt.close(fig)


def _plot_correlation(features: pd.DataFrame, target: pd.Series) -> None:
    matrix = features.copy()
    matrix["y (+1 подделка)"] = target.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix.corr(),
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "корреляция Пирсона"},
    )
    ax.set_title("Корреляции признаков и метки подделки")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_correlation.png", dpi=120)
    plt.close(fig)


def _plot_feature_scale(eda: pd.DataFrame) -> None:
    spread = eda[FEATURE_COLUMNS].std(ddof=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    spread.plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Разброс признаков до стандартизации")
    ax.set_xlabel("Признак")
    ax.set_ylabel("Стандартное отклонение")
    ax.set_xticklabels([PLOT_NAMES[column] for column in spread.index], rotation=15)
    ax.legend(["std по всей выборке"])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_feature_scale.png", dpi=120)
    plt.close(fig)


def _plot_pairplot(eda: pd.DataFrame) -> None:
    grid = sns.pairplot(
        eda,
        hue="класс",
        hue_order=[CLASS_NAMES[GENUINE], CLASS_NAMES[FORGED]],
        corner=True,
        plot_kws={"alpha": 0.55, "s": 14},
        diag_kws={"common_norm": False},
    )
    grid.figure.suptitle("Попарные проекции четырёх признаков", y=1.02)
    grid.savefig(PLOTS_DIR / "06_pairplot.png", dpi=120)
    plt.close(grid.figure)


def prepare_data(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    solver_train_size: Optional[int] = SOLVER_TRAIN_SIZE,
) -> dict:
    if df is None:
        df = load_raw_dataframe()

    y = encode_labels(df["raw_class"])
    X = df[FEATURE_COLUMNS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    full_train_size = len(X_train)

    if solver_train_size is not None and full_train_size > solver_train_size:
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=solver_train_size,
            stratify=y_train,
            random_state=random_state,
        )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
    )
    y_train = y_train.reset_index(drop=True).astype(int)
    y_test = y_test.reset_index(drop=True).astype(int)
    y_train.name = "y"
    y_test.name = "y"

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLUMNS,
        "visual_features": VISUAL_FEATURES,
        "scaler": scaler,
        "full_train_size": full_train_size,
        "label_meaning": CLASS_NAMES,
    }


def as_numpy(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data["X_train"].to_numpy(dtype=float),
        data["X_test"].to_numpy(dtype=float),
        data["y_train"].to_numpy(dtype=int),
        data["y_test"].to_numpy(dtype=int),
    )


def visual_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = data["visual_features"]
    return (
        data["X_train"][columns].to_numpy(dtype=float),
        data["X_test"][columns].to_numpy(dtype=float),
        data["y_train"].to_numpy(dtype=int),
        data["y_test"].to_numpy(dtype=int),
    )


def print_prepared_summary(data: dict) -> None:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    majority = int(y_train.value_counts().idxmax())
    baseline = float((y_test == majority).mean())

    print("\n=== После подготовки ===")
    print("признаки:", data["feature_names"])
    print("оси для рисунка границы:", data["visual_features"])
    print("метки:", data["label_meaning"])
    print(
        f"train до урезания под солвер: {data['full_train_size']}, "
        f"train для SVM: {len(X_train)}, test: {len(X_test)}"
    )
    print("метки train:", y_train.value_counts().sort_index().to_dict())
    print("метки test:", y_test.value_counts().sort_index().to_dict())
    print("пропуски train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
    print("средние train после scaler (~0):\n", X_train.mean().round(4).to_string())
    print("std train после scaler (~1):\n", X_train.std(ddof=0).round(4).to_string())
    print("средние test после scaler:\n", X_test.mean().round(4).to_string())
    print(
        f"база: всегда класс {majority} ({CLASS_NAMES[majority]}), "
        f"accuracy на test {baseline:.3f}"
    )


if __name__ == "__main__":
    raw = load_raw_dataframe()
    run_eda(raw)
    prepared = prepare_data(raw)
    print_prepared_summary(prepared)
