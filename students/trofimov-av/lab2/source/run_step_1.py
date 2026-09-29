import numpy as np

from data import load_iris_split, standardize


def main():
    x_train, x_test, y_train, y_test, feature_names, class_names = load_iris_split()
    x_train_scaled, x_test_scaled, mean, scale = standardize(x_train, x_test)

    print(f"Объектов: {len(x_train) + len(x_test)}")
    print(f"Признаков: {len(feature_names)}")
    print(f"Классы: {', '.join(class_names)}")
    print(f"Train: {x_train.shape}, распределение: {np.bincount(y_train)}")
    print(f"Test: {x_test.shape}, распределение: {np.bincount(y_test)}")
    print(f"Средние train после стандартизации: {x_train_scaled.mean(axis=0)}")
    print(f"Std train после стандартизации: {x_train_scaled.std(axis=0)}")
    print(f"Средние исходного train: {mean}")
    print(f"Std исходного train: {scale}")
    print(f"Размер стандартизованного test: {x_test_scaled.shape}")


if __name__ == "__main__":
    main()
