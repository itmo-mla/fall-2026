from pathlib import Path

import numpy as np
import pandas as pd

# Датасет Titanic (Kaggle, 891 пассажир). Хранить датасеты в репозитории нельзя,
# поэтому при первом запуске он скачивается по ссылке и сохраняется рядом с кодом.
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = Path(__file__).parent / "Titanic-Dataset.csv"

if not DATA_PATH.exists():
    pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
df = pd.read_csv(DATA_PATH)

y_map = {0: -1, 1: +1}
df['Survived'] = df['Survived'].map(y_map)

df['Pclass'] = df['Pclass'].astype('category')

gender_map = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(gender_map)

status = ['Mr', 'Miss', 'Mrs', 'Master']
status_pattern = r'\s([A-Z][a-z]+)\.'
df['Title'] = df['Name'].str.extract(status_pattern)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Age'] = df['Age'].fillna(df.groupby(['Title', 'Pclass', 'Sex'])['Age'].transform('median'))

df['Title'] = df['Title'].replace({'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss'})
mask = ~df['Title'].isin(status)
df.loc[mask, 'Title'] = 'Rare'

df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
df = pd.get_dummies(df, drop_first=True)

y = df['Survived'].copy()
X = df.drop('Survived', axis=1)
feature_names = ['bias'] + list(X.columns)  # имена признаков для вывода весов
bias_column = np.ones(len(X)).reshape(len(X), 1)
X = np.concatenate((bias_column, X), axis=1).astype(float)
y = y.to_numpy()


def train_and_test(X, y, test_size):
    test_index = np.random.choice(len(X), round(len(X)*test_size), replace=False)
    X_test = X[test_index]
    y_test = y[test_index]
    all_indices = np.arange(len(X))
    train_indices = np.setdiff1d(all_indices, test_index)
    X_train = X[train_indices]
    y_train = y[train_indices]
    return X_train, y_train, X_test, y_test

def standardize(X_train, X_test, columns):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in columns:
        mean = X_train[:, col].mean()
        std = X_train[:, col].std()
        X_train[:, col] = (X_train[:, col] - mean) / std
        X_test[:, col] = (X_test[:, col] - mean) / std
    return X_train, X_test
