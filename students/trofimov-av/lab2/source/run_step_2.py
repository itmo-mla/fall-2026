from data import load_iris_split, standardize
from parzen_knn import ParzenKNN


def main():
    x_train, x_test, y_train, y_test, _, class_names = load_iris_split()
    x_train, x_test, _, _ = standardize(x_train, x_test)

    model = ParzenKNN(k=5).fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    _, bandwidths = model.class_scores(x_test)

    for index in range(10):
        print(
            f"Объект {index + 1}: истинный={class_names[y_test[index]]}, "
            f"предсказанный={class_names[predictions[index]]}, "
            f"h={bandwidths[index]:.4f}, "
            f"веса классов={probabilities[index]}"
        )


if __name__ == "__main__":
    main()
