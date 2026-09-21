from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "student_burnout.csv"
PLOTS_DIR = ROOT_DIR / "plots"

LEAKAGE_COLUMNS = ["student_id", "burnout_score"]
TARGET_COLUMN = "high_burnout"
MISSING_COLUMNS = ["screen_time_hours", "commute_minutes", "teacher_support"]


def load_raw_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Положи CSV в lab1/ или запусти скрипт из lab1/."
        )
    return pd.read_csv(path)

def run_eda(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    print(df.info())
    print("\n=== Пропуски ===")
    print(df.isna().sum())
    print("\n=== Статистика ===")
    print(df.describe())
    print("\n=== Значения gender ===")
    print(df["gender"].value_counts())
    print("\n=== Баланс high_burnout ===")
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    print("\n=== Утечка: burnout_score vs high_burnout ===")
    print(pd.crosstab(df["burnout_score"], df[TARGET_COLUMN]))
    print("корреляция:", df["burnout_score"].corr(df[TARGET_COLUMN]))
    print(
        "Вывод: high_burnout == 1 ровно тогда, когда burnout_score >= 4. "
        "Это не признак, а тот же таргет в другом виде."
    )

    fig, ax = plt.subplots()
    df[TARGET_COLUMN].value_counts().sort_index().plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Баланс классов high_burnout")
    ax.set_xlabel("Класс (0 — нет выгорания, 1 — есть)")
    ax.set_ylabel("Число студентов")
    ax.legend(["число объектов"])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="sleep_hours", hue=TARGET_COLUMN, bins=20, ax=ax, kde=True)
    ax.set_title("Часы сна в разрезе класса выгорания")
    ax.set_xlabel("Часы сна")
    ax.set_ylabel("Число студентов")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_sleep_hours_by_class.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="sleep_quality", hue=TARGET_COLUMN, bins=5, ax=ax, discrete=True)
    ax.set_title("Качество сна в разрезе класса выгорания")
    ax.set_xlabel("Качество сна")
    ax.set_ylabel("Число студентов")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_sleep_quality_by_class.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=TARGET_COLUMN, y="screen_time_hours", ax=ax)
    ax.set_title("Экранное время по классам")
    ax.set_xlabel("high_burnout")
    ax.set_ylabel("Часы у экрана")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_screen_time_boxplot.png", dpi=120)
    plt.close(fig)

    numeric_for_corr = df.drop(columns=LEAKAGE_COLUMNS + [TARGET_COLUMN], errors="ignore")
    numeric_for_corr = numeric_for_corr.select_dtypes(include="number")
    corr = numeric_for_corr.join(df[TARGET_COLUMN]).corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
    ax.set_title("Корреляции признаков (без id и burnout_score)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_correlation_heatmap.png", dpi=120)
    plt.close(fig)


def prepare_data(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    if df is None:
        df = load_raw_dataframe()

    y = (2 * df[TARGET_COLUMN] - 1).astype(int)#переводим таргет к -1/+1
    X = df.drop(columns=LEAKAGE_COLUMNS + [TARGET_COLUMN]) #убираем ненужные колонки типо айди студентов и схожих с таргетов
    X = pd.get_dummies(X, columns=["gender"], dtype=int, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y,)
    medians = X_train[MISSING_COLUMNS].median() #Для пропусков сетим медианне значения
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_train_scaled.columns),
        "scaler": scaler,
        "medians": medians,
    }


def print_prepared_summary(data: dict) -> None:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("\n=== После подготовки ===")
    print("признаки:", data["feature_names"])
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("метки train:", y_train.value_counts().to_dict())
    print("метки test:", y_test.value_counts().to_dict())
    print("пропуски train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
    print("средние train после scaler (должны быть ~0):\n", X_train.mean().round(4).to_string())
    print("std train после scaler (должны быть ~1):\n", X_train.std(ddof=0).round(4).to_string())
    print("средние test после scaler (близко к 0, но не идеально):\n", X_test.mean().round(4).to_string())


if __name__ == "__main__":
    raw_df = load_raw_dataframe()
    run_eda(raw_df)
    prepared = prepare_data(raw_df)
    print_prepared_summary(prepared)
    print(f"\nГрафики сохранены в {PLOTS_DIR}")
