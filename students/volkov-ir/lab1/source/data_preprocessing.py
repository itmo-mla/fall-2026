from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = Path(__file__).parent / "Titanic-Dataset.csv"

# Допустимые значения категориальных признаков заданы заранее, а не берутся из данных
CATEGORIES = {
    'Pclass': [1, 2, 3],
    'Embarked': ['C', 'Q', 'S'],
    'Title': ['Master', 'Miss', 'Mr', 'Mrs', 'Rare'],
}
TITLE_FIX = {'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss'}
TITLE_KEEP = ['Mr', 'Miss', 'Mrs', 'Master']


def load_data():
    # Только преобразования, которые считаются по каждой строке отдельно
    if not DATA_PATH.exists():
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    df['Survived'] = df['Survived'].map({0: -1, 1: +1})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Title'] = df['Name'].str.extract(r'\s([A-Z][a-z]+)\.')
    return df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])


def train_and_test(df, test_size):
    # Разбиение до любых статистик, чтобы test не влиял на обучение
    test_index = np.random.choice(len(df), round(len(df) * test_size), replace=False)
    train_index = np.setdiff1d(np.arange(len(df)), test_index)
    return df.iloc[train_index], df.iloc[test_index]


def fit_imputer(df_train):
    # Статистики для заполнения пропусков, считаются только по train
    return {
        'embarked': df_train['Embarked'].mode()[0],
        'age_group': df_train.groupby(['Title', 'Pclass', 'Sex'])['Age'].median(),
        'age_class_sex': df_train.groupby(['Pclass', 'Sex'])['Age'].median(),
        'age_all': df_train['Age'].median(),
    }


def apply_imputer(df, stats):
    # Заполнение пропусков статистиками из train, одинаково для train и test
    df = df.copy()
    df['Embarked'] = df['Embarked'].fillna(stats['embarked'])
    group_key = pd.MultiIndex.from_frame(df[['Title', 'Pclass', 'Sex']])
    class_key = pd.MultiIndex.from_frame(df[['Pclass', 'Sex']])
    age_group = pd.Series(stats['age_group'].reindex(group_key).to_numpy(), index=df.index)
    age_class = pd.Series(stats['age_class_sex'].reindex(class_key).to_numpy(), index=df.index)
    # Если группы не было в train, берется более грубая группа, затем общая медиана
    df['Age'] = df['Age'].fillna(age_group).fillna(age_class).fillna(stats['age_all'])
    return df


def build_features(df):
    # One-hot и столбец bias, считается по каждой строке отдельно
    df = df.copy()
    title = df['Title'].replace(TITLE_FIX)
    df['Title'] = title.where(title.isin(TITLE_KEEP), 'Rare')
    for column, categories in CATEGORIES.items():
        df[column] = pd.Categorical(df[column], categories=categories)
    df = pd.get_dummies(df, drop_first=True)
    y = df.pop('Survived').to_numpy()
    X = np.column_stack([np.ones(len(df)), df.to_numpy(dtype=float)])
    return X, y, ['bias'] + list(df.columns)


def prepare_data(test_size):
    # Полная цепочка, порядок важен, разбиение идет раньше всех статистик
    df_train, df_test = train_and_test(load_data(), test_size)
    stats = fit_imputer(df_train)
    X_train, y_train, feature_names = build_features(apply_imputer(df_train, stats))
    X_test, y_test, _ = build_features(apply_imputer(df_test, stats))
    return X_train, y_train, X_test, y_test, feature_names


def standardize(X_train, X_test, columns):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in columns:
        mean = X_train[:, col].mean()
        std = X_train[:, col].std()
        X_train[:, col] = (X_train[:, col] - mean) / std
        X_test[:, col] = (X_test[:, col] - mean) / std
    return X_train, X_test
