"""Загрузка и подготовка данных для лабораторной работы № 5."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(test_size=0.2, random_state=42):
    """Подготовить Wisconsin Breast Cancer для бинарной классификации.

    Датасет поставляется со scikit-learn и не требует скачивания.
    Возвращает X_train, X_test, y_train, y_test. Параметры стандартизации
    оцениваются только на обучающей выборке, чтобы избежать утечки данных.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    return X_train, X_test, y_train, y_test
