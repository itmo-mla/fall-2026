import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "plots"

KAGGLE_DATASET = "natezhang123/social-anxiety-dataset"
DATA_FILE = "enhanced_anxiety_dataset.csv"

TARGET_COLUMN = "Anxiety Level (1-10)"
TARGET_NAME = "anxiety_level"

NUMERIC_COLUMNS = [
    "Age",
    "Sleep Hours",
    "Physical Activity (hrs/week)",
    "Caffeine Intake (mg/day)",
    "Alcohol Consumption (drinks/week)",
    "Diet Quality (1-10)",
    "Stress Level (1-10)",
    "Heart Rate (bpm)",
    "Breathing Rate (breaths/min)",
    "Sweating Level (1-5)",
    "Therapy Sessions (per month)",
]
BINARY_COLUMNS = [
    "Smoking",
    "Family History of Anxiety",
    "Dizziness",
    "Medication",
    "Recent Major Life Event",
]
GENDER_LEVELS = ["Female", "Male", "Other"]
OCCUPATION_REFERENCE = "Other"

SHORT_NAMES = {
    "Age": "age",
    "Sleep Hours": "sleep_hours",
    "Physical Activity (hrs/week)": "physical_activity",
    "Caffeine Intake (mg/day)": "caffeine_mg",
    "Alcohol Consumption (drinks/week)": "alcohol_drinks",
    "Diet Quality (1-10)": "diet_quality",
    "Stress Level (1-10)": "stress_level",
    "Heart Rate (bpm)": "heart_rate",
    "Breathing Rate (breaths/min)": "breathing_rate",
    "Sweating Level (1-5)": "sweating_level",
    "Therapy Sessions (per month)": "therapy_sessions",
    "Smoking": "smoking",
    "Family History of Anxiety": "family_history",
    "Dizziness": "dizziness",
    "Medication": "medication",
    "Recent Major Life Event": "major_life_event",
}
PLOT_NAMES = {
    "Age": "возраст, лет",
    "Sleep Hours": "сон, часы",
    "Physical Activity (hrs/week)": "активность, ч/нед",
    "Caffeine Intake (mg/day)": "кофеин, мг/день",
    "Alcohol Consumption (drinks/week)": "алкоголь, порций/нед",
    "Diet Quality (1-10)": "диета, 1–10",
    "Stress Level (1-10)": "стресс, 1–10",
    "Heart Rate (bpm)": "пульс, уд/мин",
    "Breathing Rate (breaths/min)": "дыхание, вдохов/мин",
    "Sweating Level (1-5)": "потливость, 1–5",
    "Therapy Sessions (per month)": "терапия, сессий/мес",
    "Smoking": "курение",
    "Family History of Anxiety": "тревожность в семье",
    "Dizziness": "головокружение",
    "Medication": "лекарства",
    "Recent Major Life Event": "крупное событие",
}
STRONG_FEATURES = [
    "Stress Level (1-10)",
    "Sleep Hours",
    "Caffeine Intake (mg/day)",
    "Therapy Sessions (per month)",
]
BOX_COLOR = "#4C72B0"
EXPECTED_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + [
    "Gender",
    "Occupation",
    TARGET_COLUMN,
]


def _download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Рядом со скриптом нет CSV. Для скачивания нужен пакет kagglehub: pip install kagglehub"
        ) from exc

    directory = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    path = directory / DATA_FILE
    if not path.exists():
        found = sorted(item.name for item in directory.glob("*.csv"))
        raise FileNotFoundError(f"В {directory} нет {DATA_FILE}. Нашлись: {found}")
    return path


def load_raw_dataframe() -> pd.DataFrame:
    local = ROOT_DIR / DATA_FILE
    if local.exists():
        frame = pd.read_csv(local)
    else:
        frame = pd.read_csv(_download_dataset())
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing or frame.shape[1] != len(EXPECTED_COLUMNS):
        raise RuntimeError(
            "Колонки файла не совпали с версией, на которой сделан EDA. "
            f"Сейчас: {list(frame.columns)}"
        )
    return frame


def _occupation_levels(occupations: pd.Series) -> list[str]:
    observed = [str(level) for level in occupations.unique()]
    if OCCUPATION_REFERENCE not in observed:
        raise RuntimeError(f"Нет опорной профессии {OCCUPATION_REFERENCE!r}: {sorted(observed)}")
    rest = sorted(level for level in observed if level != OCCUPATION_REFERENCE)
    return [OCCUPATION_REFERENCE, *rest]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Числовая матрица признаков без цели.

    Yes/No кодируются как 1/0. Пол и профессия — one-hot без первого уровня:
    опорные категории Female и Other. Так в столбцах нет точной суммы «единица»,
    и гребневая регрессия потом не упирается в вырожденную матрицу.
    """
    pieces: list[pd.DataFrame] = []

    numeric = df[NUMERIC_COLUMNS].astype(float).rename(columns=SHORT_NAMES)
    pieces.append(numeric)

    binary = pd.DataFrame(
        {SHORT_NAMES[column]: (df[column] == "Yes").astype(float) for column in BINARY_COLUMNS},
        index=df.index,
    )
    unexpected = {
        column: sorted(set(df[column].unique()) - {"Yes", "No"})
        for column in BINARY_COLUMNS
        if set(df[column].unique()) - {"Yes", "No"}
    }
    if unexpected:
        raise RuntimeError(f"В бинарных колонках не только Yes/No: {unexpected}")
    pieces.append(binary)

    if set(df["Gender"].unique()) - set(GENDER_LEVELS):
        raise RuntimeError(f"Неизвестный пол: {sorted(df['Gender'].unique())}")
    gender = pd.Series(
        pd.Categorical(df["Gender"], categories=GENDER_LEVELS),
        index=df.index,
        name="gender",
    )
    pieces.append(pd.get_dummies(gender, prefix="gender", drop_first=True, dtype=float))

    occupation_levels = _occupation_levels(df["Occupation"])
    occupation = pd.Series(
        pd.Categorical(df["Occupation"], categories=occupation_levels),
        index=df.index,
        name="occupation",
    )
    pieces.append(pd.get_dummies(occupation, prefix="occupation", drop_first=True, dtype=float))

    matrix = pd.concat(pieces, axis=1)
    if matrix.columns.duplicated().any():
        raise RuntimeError(f"Повторяющиеся столбцы: {matrix.columns[matrix.columns.duplicated()].tolist()}")
    return matrix


def run_eda(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    numeric = df[NUMERIC_COLUMNS + [TARGET_COLUMN]]

    print("\n=== Размер и пропуски ===")
    print(f"объектов: {len(df)}, колонок в файле: {df.shape[1]}")
    print(f"пропусков: {int(df.isna().sum().sum())}, полных дубликатов строк: {int(df.duplicated().sum())}")

    print("\n=== Цель: уровень тревожности ===")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())
    print(df[TARGET_COLUMN].describe().round(3).to_string())

    print("\n=== Категории ===")
    for column in ["Gender", "Occupation", *BINARY_COLUMNS]:
        print(column, df[column].value_counts().to_dict())

    print("\n=== Std числовых признаков до масштаба (ddof=0) ===")
    spread = numeric.drop(columns=TARGET_COLUMN).std(ddof=0).sort_values(ascending=False)
    print(spread.round(3).to_string())

    print("\n=== Корреляция Пирсона с уровнем тревожности ===")
    pearson = numeric.drop(columns=TARGET_COLUMN).corrwith(df[TARGET_COLUMN]).sort_values()
    print(pearson.round(3).to_string())
    print("\n=== Корреляция Спирмена с уровнем тревожности ===")
    spearman = numeric.drop(columns=TARGET_COLUMN).corrwith(df[TARGET_COLUMN], method="spearman").sort_values()
    print(spearman.round(3).to_string())

    feature_corr = numeric.drop(columns=TARGET_COLUMN).corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    print("\n=== Максимальная |корреляция| между признаками, не с целью ===")
    flat = upper.stack().sort_values(ascending=False)
    print(flat.head(5).round(3).to_string())

    print("\n=== Средняя тревожность по бинарным факторам ===")
    for column in BINARY_COLUMNS:
        means = df.groupby(column)[TARGET_COLUMN].mean().round(3)
        print(column)
        print(means.to_string())

    print("\n=== Средняя тревожность по полу и профессии ===")
    print(df.groupby("Gender")[TARGET_COLUMN].mean().round(3).to_string())
    occupation_means = df.groupby("Occupation")[TARGET_COLUMN].mean().sort_values()
    print(occupation_means.round(3).to_string())
    print(
        "размах средних по профессии:",
        round(float(occupation_means.max() - occupation_means.min()), 3),
    )

    print("\n=== Выбросы по правилу 1.5 IQR (не удаляем) ===")
    for column in numeric.columns:
        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1
        mask = (numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)
        print(f"{column}: {int(mask.sum())} ({mask.mean():.1%})")

    _plot_target(df)
    _plot_feature_scale(df)
    _plot_correlation(numeric)
    _plot_strong_features(df)
    _plot_binary_means(df)
    _plot_occupation(df, occupation_means)
    print(f"\nГрафики: {PLOTS_DIR}")


def _plot_target(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = pd.DataFrame({"балл": counts.index.astype(int).astype(str), "число людей": counts.to_numpy()})
    sns.barplot(data=plot_df, x="балл", y="число людей", color=BOX_COLOR, ax=ax)
    ax.set_title("Распределение уровня социальной тревожности")
    ax.set_xlabel("Уровень тревожности, балл от 1 до 10")
    ax.set_ylabel("Число людей")
    ax.legend(handles=[Patch(facecolor=BOX_COLOR, label="число людей с этим баллом")])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_anxiety_distribution.png", dpi=120)
    plt.close(fig)


def _plot_feature_scale(df: pd.DataFrame) -> None:
    spread = df[NUMERIC_COLUMNS].std(ddof=0).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [PLOT_NAMES[column] for column in spread.index]
    ax.barh(labels, spread.to_numpy(), color=BOX_COLOR, label="std по всей выборке")
    ax.set_title("Разброс признаков до стандартизации")
    ax.set_xlabel("Стандартное отклонение в исходных единицах")
    ax.set_ylabel("Признак")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_feature_scale.png", dpi=120)
    plt.close(fig)


def _plot_correlation(numeric: pd.DataFrame) -> None:
    renamed = numeric.rename(columns={**PLOT_NAMES, TARGET_COLUMN: "тревожность, 1–10"})
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        renamed.corr(),
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "корреляция Пирсона"},
    )
    ax.set_title("Корреляции числовых признаков и уровня тревожности")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_correlation.png", dpi=120)
    plt.close(fig)


def _plot_strong_features(df: pd.DataFrame) -> None:
    plot_df = df[STRONG_FEATURES + [TARGET_COLUMN]].copy()
    plot_df["балл тревожности"] = plot_df[TARGET_COLUMN].astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, column in zip(axes.ravel(), STRONG_FEATURES):
        sns.boxplot(
            data=plot_df,
            x="балл тревожности",
            y=column,
            color=BOX_COLOR,
            fliersize=2,
            ax=ax,
        )
        ax.set_title(PLOT_NAMES[column])
        ax.set_xlabel("Уровень тревожности, балл")
        ax.set_ylabel(PLOT_NAMES[column])
        ax.legend(handles=[Patch(facecolor=BOX_COLOR, label="ящик и усы 1.5 IQR")])
    fig.suptitle("Признаки с самой сильной связью с тревогой")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_strong_features.png", dpi=120)
    plt.close(fig)


def _plot_binary_means(df: pd.DataFrame) -> None:
    rows = []
    for column in BINARY_COLUMNS:
        for answer, mean in df.groupby(column)[TARGET_COLUMN].mean().items():
            rows.append(
                {
                    "фактор": PLOT_NAMES[column],
                    "ответ": {"No": "нет", "Yes": "да"}[str(answer)],
                    "средняя тревожность": mean,
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=plot_df,
        x="фактор",
        y="средняя тревожность",
        hue="ответ",
        hue_order=["нет", "да"],
        ax=ax,
    )
    ax.set_ylim(1, 10)
    ax.set_title("Средняя тревожность при бинарных факторах, шкала 1–10 целиком")
    ax.set_xlabel("Фактор")
    ax.set_ylabel("Средний балл тревожности")
    ax.legend(title="Ответ")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_binary_factors.png", dpi=120)
    plt.close(fig)


def _plot_occupation(df: pd.DataFrame, occupation_means: pd.Series) -> None:
    overall = float(df[TARGET_COLUMN].mean())
    gap = float(occupation_means.max() - occupation_means.min())
    ordered = occupation_means.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordered.index, ordered.to_numpy() - overall, color=BOX_COLOR, label="отклонение среднего профессии")
    ax.axvline(0, color="#C44E52", linewidth=1.2, label="среднее по всей выборке")
    ax.set_title(f"Профессия сдвигает среднюю тревожность максимум на {gap:.2f} балла")
    ax.set_xlabel("Отклонение от среднего балла тревожности по выборке")
    ax.set_ylabel("Профессия")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_occupation.png", dpi=120)
    plt.close(fig)


def prepare_data(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Матрица для PCA и линейной регрессии.

    StandardScaler обучается только на train. Цель в матрицу не входит.
    """
    if df is None:
        df = load_raw_dataframe()

    y = df[TARGET_COLUMN].astype(float)
    y.name = TARGET_NAME
    X = design_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y.astype(int),
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
    )
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train.name = TARGET_NAME
    y_test.name = TARGET_NAME

    if not np.allclose(X_train_scaled.mean().to_numpy(), 0.0, atol=1e-8):
        raise RuntimeError("После StandardScaler среднее train не ноль: scaler обучен не на train.")
    if not np.allclose(X_train_scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-6):
        raise RuntimeError("После StandardScaler std train (ddof=0) не единица.")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "reference_levels": {"gender": "Female", "occupation": OCCUPATION_REFERENCE},
    }


def as_numpy(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Массивы float для SVD. Цель не целочисленная: балл тревожности — это регрессия."""
    return (
        data["X_train"].to_numpy(dtype=float),
        data["X_test"].to_numpy(dtype=float),
        data["y_train"].to_numpy(dtype=float),
        data["y_test"].to_numpy(dtype=float),
    )


def print_prepared_summary(data: dict) -> None:
    X_train = data["X_train"]
    X_test = data["X_test"]
    print("\n=== После подготовки ===")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print("опорные уровни one-hot:", data["reference_levels"])
    print("столбцы:")
    print("\n".join(data["feature_names"]))
    print("пропуски train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
    print("max |среднее| train:", float(np.max(np.abs(X_train.mean()))))
    print("std train ddof=0, min/max:", float(X_train.std(ddof=0).min()), float(X_train.std(ddof=0).max()))
    print("max |среднее| test:", round(float(np.max(np.abs(X_test.mean()))), 4))
    print("цель train, среднее/std:", round(float(data["y_train"].mean()), 3), round(float(data["y_train"].std()), 3))
    print("цель test, среднее/std:", round(float(data["y_test"].mean()), 3), round(float(data["y_test"].std()), 3))
    print("баллы train:", data["y_train"].astype(int).value_counts().sort_index().to_dict())
    print("баллы test:", data["y_test"].astype(int).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    raw = load_raw_dataframe()
    run_eda(raw)
    prepared = prepare_data(raw)
    print_prepared_summary(prepared)
