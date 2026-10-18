import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classifier import LinearClassifier
from data import (
    FEATURE_COLUMNS,
    describe_dataset,
    load_dataset,
    prepare_data,
    split_and_scale,
)
from metric import calculate_confusion_matrix, evaluate_predictions
from sklearn.linear_model import RidgeClassifier
from visualization import (
    analyze_margins,
    calculate_correlations,
    plot_confusion_matrix,
    plot_margins,
    plot_quality_history,
)

SGD_PARAMS = {
    "learning_rate": 0.001,
    "beta": 0.9,
    "l2": 0.001,
    "alpha": 0.01,
    "epochs": 50,
}


def run_multistart(X_train, y_train, n_starts=10):
    """Обучает n_starts моделей и выбирает минимум train loss с L2.

    Возвращает лучшую модель и DataFrame результатов всех запусков.
    В каждом запуске seed управляет и весами, и порядком объектов.
    """
    best_model = None
    best_train_loss = np.inf
    results = []
    for seed in range(n_starts):
        model = LinearClassifier(
            **SGD_PARAMS, random_state=seed, init_method="random"
        ).fit(X_train, y_train)
        train_loss = model.mean_loss(X_train, y_train, regularized=True)
        results.append({"seed": seed, "train_loss": train_loss})
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model
    return best_model, pd.DataFrame(results)


def print_dataset_summary(df):
    """Печатает первые строки и описательную статистику датасета."""
    summary = describe_dataset(df)
    print(df.head().to_string(index=False))
    for label, key in [
        ("Размер датасета", "shape"),
        ("Типы данных", "dtypes"),
        ("Пропуски", "missing"),
        ("Распределение классов", "class_counts"),
        ("Описательная статистика", "description"),
    ]:
        print(f"\n{label}:\n{summary[key]}")


def compare_with_reference(
    best_model_name, our_predictions, X_train, y_train, X_test, y_test
):
    """Сравнивает готовые предсказания лучшей модели с RidgeClassifier.

    Возвращает таблицу метрик, признак совпадения ответов и матрицу ошибок
    эталона. Его настройки и обучающая выборка соответствуют экспериментам.
    """
    reference_model = RidgeClassifier(alpha=0.001).fit(X_train, y_train)
    reference_predictions = reference_model.predict(X_test)
    comparison = pd.DataFrame(
        [
            {
                "model": f"Our: {best_model_name}",
                **evaluate_predictions(y_test, our_predictions),
            },
            {
                "model": "RidgeClassifier",
                **evaluate_predictions(y_test, reference_predictions),
            },
        ]
    )
    same_predictions = np.array_equal(our_predictions, reference_predictions)
    reference_matrix = calculate_confusion_matrix(y_test, reference_predictions)
    return comparison, same_predictions, reference_matrix


def run_experiments():
    """Скачивает данные, выполняет все эксперименты и строит графики.

    Возвращает модели, итоговую таблицу, результаты мультистарта,
    статистику отступов и сравнение с эталонным RidgeClassifier.
    Настройки и выбор моделей соответствуют исходному ноутбуку.
    """
    print("1. Загрузка Banknote Authentication из UCI")
    df = load_dataset()
    print_dataset_summary(df)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"\nTrain: {X_train.shape}; Test: {X_test.shape}")
    print("Средние train после стандартизации:", X_train.mean(axis=0))
    print("Стандартные отклонения train:", X_train.std(axis=0))
    print("Число объектов, использованных scaler:", scaler.n_samples_seen_)
    print(
        f"SGD: {SGD_PARAMS['epochs']} эпох, "
        f"{SGD_PARAMS['epochs'] * len(y_train)} шагов на модель."
    )

    correlations = calculate_correlations(X_train, y_train, FEATURE_COLUMNS)
    print("\nКорреляции признаков с меткой:\n", correlations)

    print("\n2. Корреляционная инициализация")
    models = {
        "Correlation init": LinearClassifier(
            **SGD_PARAMS, random_state=42, init_method="correlation"
        ).fit(X_train, y_train)
    }

    print("3. Случайная инициализация: мультистарт из 10 запусков")
    models["Multistart"], multistart_results = run_multistart(X_train, y_train)
    print(multistart_results.to_string(index=False))
    print("Лучший seed:", models["Multistart"].random_state)

    print("4. Сравнение случайного порядка и сортировки по модулю отступа")
    for name, order in [("Random order", "random"), ("Margin order", "margin")]:
        models[name] = LinearClassifier(
            **SGD_PARAMS, random_state=42, init_method="random", sample_order=order
        ).fit(X_train, y_train)

    print("5. Скорейший градиентный спуск: максимум 100 итераций")
    models["Steepest"] = LinearClassifier(
        l2=0.001,
        alpha=0.1,
        epochs=100,
        random_state=42,
        optimizer="steepest",
        init_method="random",
    ).fit(X_train, y_train)

    results = []
    margin_rows = []
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        scores = evaluate_predictions(y_test, predictions[name])
        results.append(
            {
                "model": name,
                **scores,
                "test_loss": model.mean_loss(X_test, y_test, regularized=False),
            }
        )
        margins = model.margin(X_test, y_test)
        margin_rows.append({"model": name, **analyze_margins(margins)})
        matrix = calculate_confusion_matrix(y_test, predictions[name])
        print(f"\n{name}:\nw = {model.w}\nb = {model.b}")
        print("Количество оценок Q:", len(model.quality_history))
        print("Матрица ошибок:\n", matrix)

        plot_margins(margins, f"Распределение отступов — {name}")
        plot_quality_history(
            {name: model.quality_history},
            f"Процесс обучения — {name}",
            xlabel="Итерация полного градиента" if name == "Steepest" else "Шаг SGD",
        )
        plot_confusion_matrix(matrix, f"Матрица ошибок — {name}")

    plot_quality_history(
        {name: models[name].quality_history for name in ("Random order", "Margin order")},
        "Влияние порядка предъявления объектов на обучение",
    )
    results_df = pd.DataFrame(results)
    margin_summary = pd.DataFrame(margin_rows)
    print("\n6. Сравнение моделей:\n", results_df.to_string(index=False))
    print("\nХарактеристики отступов:\n", margin_summary.to_string(index=False))
    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_model_name = best_row["model"]

    print("\n7. Сравнение лучшей модели с RidgeClassifier")
    comparison, same_predictions, reference_matrix = compare_with_reference(
        best_model_name, predictions[best_model_name], X_train, y_train, X_test, y_test
    )
    print(comparison.to_string(index=False))
    print("Предсказания полностью совпадают:", same_predictions)
    plot_confusion_matrix(
        reference_matrix,
        "Матрица ошибок — RidgeClassifier",
    )
    return {
        "models": models,
        "results": results_df,
        "multistart_results": multistart_results,
        "margin_summary": margin_summary,
        "comparison": comparison,
        "best_model_name": best_model_name,
        "same_predictions": same_predictions,
    }


if __name__ == "__main__":
    run_experiments()
    plt.show()
