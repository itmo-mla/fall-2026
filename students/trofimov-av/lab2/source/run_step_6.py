import numpy as np

from data import load_iris_split, standardize
from parzen_knn import ParzenKNN
from prototype_selection import select_prototypes


def main():
    x_train, x_test, y_train, _, _, class_names = load_iris_split()
    x_train, _, _, _ = standardize(x_train, x_test)
    result = select_prototypes(x_train, y_train, k=1, noise_threshold=0.0)

    prototype_model = ParzenKNN(k=1).fit(
        x_train[result.prototype_indices], y_train[result.prototype_indices]
    )
    retained_mask = np.ones(len(y_train), dtype=bool)
    retained_mask[result.noise_indices] = False
    retained_predictions = prototype_model.predict(x_train[retained_mask])
    retained_accuracy = np.mean(retained_predictions == y_train[retained_mask])

    print(f"Исходных объектов: {len(y_train)}")
    print(f"Выбрано эталонов: {len(result.prototype_indices)}")
    print(f"Выделено шумовых объектов: {len(result.noise_indices)}")
    print(f"Сжатие: {1 - len(result.prototype_indices) / len(y_train):.2%}")
    print(f"Accuracy на очищенном train: {retained_accuracy:.2%}")
    print(f"История числа ошибок: {result.error_history}")
    for class_index, class_name in enumerate(class_names):
        prototype_count = np.count_nonzero(y_train[result.prototype_indices] == class_index)
        noise_count = np.count_nonzero(y_train[result.noise_indices] == class_index)
        print(f"{class_name}: эталонов={prototype_count}, шумовых={noise_count}")


if __name__ == "__main__":
    main()
