"""Эксперименты лабораторной работы № 2 в порядке пунктов README.md."""

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from data import DataProcessor
from parzen import ParzenClassifier
from prototypes import PrototypeSelector
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from visualization import Visualizer

DEFAULT_DATASET_ID = "fedesoriano/stellar-classification-dataset-sdss17"
OUTPUT_DIR = Path(__file__).with_name("figures")
EXCLUDED_COLUMNS = (
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "fiber_ID",
    "plate",
)
RANDOM_STATE = 42
TRAIN_FRACTION = 0.7
K_VALUES = tuple(range(1, 32))
# Размеры подвыборок соответствуют KNN.ipynb.
LOO_SAMPLE_SIZE = 1000
PROJECTION_TRAIN_SIZE = 1000
PROJECTION_TEST_SIZE = 1000
SELECTION_TRAIN_SIZE = 2500 // 10
SELECTION_TEST_SIZE = 5000
CCV_CONTROL_SIZE = 2
TIMING_REPEATS = 1
# В GIF ограничены только показанные шаги; полная история сохраняется в CSV.
SELECTION_GIF_MAX_STEPS = 20
# Допуски одного шага соразмерны изменению доли ошибок на одном объекте.
# REMOVAL_TOLERANCES = (
#     0.0,
#     0.5 / SELECTION_TRAIN_SIZE,
#     1.0 / SELECTION_TRAIN_SIZE,
# )
BASELINE_REMOVAL_TOLERANCE = 0.001
REMOVAL_TOLERANCES = [
    0.0,
    0.0005,
    3.985e-4,
    BASELINE_REMOVAL_TOLERANCE,
]


def make_parzen(k: int) -> ParzenClassifier:
    """Пункт 2: реализовать алгоритм KNN с методом окна Парзена переменной ширины;
                в качестве ядра можно использовать гауссово ядро;
    k задаёт ширину окна через (k + 1)-го соседа. Голосуют все опорные объекты.
    """
    return ParzenClassifier(
        window="variable", k=k, kernel=ParzenClassifier.gaussian_kernel
    )


def select_k_loo(
    X_raw: np.ndarray,
    y: np.ndarray,
    k_values: Sequence[int],
) -> tuple[int, int, pd.DataFrame]:
    """Пункт 3: подобрать параметр k методом скользящего контроля (LOO);

    Масштабирование рассчитывается только по обучающей части каждого разбиения.
    При равном риске выбрать меньшее k. Таблица нужна для графика в пункте 4.
    """
    k_values = sorted(set(k_values))
    if not k_values or any(
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or not 1 <= k < len(X_raw) - 1
        for k in k_values
    ):
        raise ValueError("В LOO для Парзена требуется 1 <= k <= len(X_raw) - 2")

    parzen_errors = np.zeros(len(k_values), dtype=int)
    sklearn_errors = np.zeros(len(k_values), dtype=int)
    for validation_index in range(len(X_raw)):
        train_mask = np.arange(len(X_raw)) != validation_index
        X_fit, X_validation = DataProcessor.standardize_pair(
            X_raw[train_mask], X_raw[validation_index : validation_index + 1]
        )
        y_fit = y[train_mask]
        for position, k in enumerate(k_values):
            parzen = make_parzen(k).fit(X_fit, y_fit)
            sklearn_knn = KNeighborsClassifier(n_neighbors=k).fit(X_fit, y_fit)
            parzen_errors[position] += (
                parzen.predict_one(X_validation[0]) != y[validation_index]
            )
            sklearn_errors[position] += (
                sklearn_knn.predict(X_validation)[0] != y[validation_index]
            )

    results = pd.DataFrame(
        {
            "k": k_values,
            "parzen_error": parzen_errors / len(X_raw),
            "sklearn_error": sklearn_errors / len(X_raw),
        }
    )
    k_parzen = int(results.loc[results["parzen_error"].idxmin(), "k"])
    k_sklearn = int(results.loc[results["sklearn_error"].idxmin(), "k"])
    return k_parzen, k_sklearn, results


def evaluate_classifier(
    classifier: ParzenClassifier | KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    repeats: int = TIMING_REPEATS,
) -> dict[str, object]:
    """Пункт 5: оценить обученную модель на test по всем признакам.

    Время predict измеряется repeats раз; возвращается медиана.
    Время обучения и отбора эталонов сюда не входит.
    """
    if len(y_test) == 0 or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("Нужны непустой test и положительное число repeats")
    times = []
    for _ in range(repeats):
        started = perf_counter()
        predictions = classifier.predict(X_test)
        times.append(perf_counter() - started)

    classes = np.unique(np.concatenate((classifier.classes_, y_test)))
    recalls = recall_score(
        y_test, predictions, labels=classes, average=None, zero_division=0
    )
    metrics = {
        "n_test": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(
                y_test, predictions, labels=classes, average="macro", zero_division=0
            )
        ),
        "error_count": int(np.count_nonzero(predictions != y_test)),
        "predict_seconds": float(np.median(times)),
    }
    for label, recall in zip(classes, recalls):
        metrics[f"recall_{label}"] = float(recall)
    return metrics


def compare_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k_parzen: int,
    k_sklearn: int,
) -> pd.DataFrame:
    """Пункт 5: сравнить Парзен и sklearn KNN на одинаковых train/test.

    Для каждого классификатора использовать его k, выбранное по LOO.
    """
    models = [
        ("Парзен", make_parzen(k_parzen), k_parzen),
        ("sklearn KNN", KNeighborsClassifier(n_neighbors=k_sklearn), k_sklearn),
    ]
    rows = []
    for name, model, k in models:
        model.fit(X_train, y_train)
        rows.append(
            {
                "name": name,
                "k": k,
                "n_reference": len(X_train),
                **evaluate_classifier(model, X_test, y_test, repeats=TIMING_REPEATS),
            }
        )
    return pd.DataFrame(rows)


def select_removal_tolerance(
    selector: PrototypeSelector,
    tolerances: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    tolerances = sorted(set(tolerances))
    if not tolerances:
        raise ValueError("Нужен хотя бы один допуск удаления")
    rows = []
    for tolerance in tolerances:
        indices, history = selector.greedy_removal(tolerance=tolerance)
        rows.append(
            {
                "tolerance": float(tolerance),
                "n_prototypes": len(indices),
                "ccv": history[-1]["ccv"],
            }
        )
    results = pd.DataFrame(rows)
    best = results.sort_values(["ccv", "n_prototypes", "tolerance"]).iloc[0]
    return float(best["tolerance"]), results


def evaluate_selection_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history: list[dict[str, object]],
    *,
    k: int,
) -> np.ndarray:

    accuracy = np.full(len(history), np.nan)
    for position, entry in enumerate(history):
        indices = np.asarray(entry["indices"])
        if len(indices) < k + 1:
            continue
        model = make_parzen(k).fit(X_train[indices], y_train[indices])
        accuracy[position] = accuracy_score(y_test, model.predict(X_test))
    return accuracy


def compare_support_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int,
    subsets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Пункт 8: сравнить качество работы KNN с и без отбора эталонов;

    Меняется только опорный набор. k, ядро, масштабирование и test одинаковы.
    Если эталонов меньше k + 1, отметить невозможность расчёта без изменения k.
    """
    rows = []
    for name, indices in subsets.items():
        indices = np.asarray(indices)
        row = {
            "name": name,
            "k": k,
            "n_reference": len(indices),
            "n_test": len(y_test),
        }
        if len(indices) < k + 1:
            row.update(
                {
                    "status": f"Нужно минимум {k + 1} опорных объектов",
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "error_count": np.nan,
                    "predict_seconds": np.nan,
                }
            )
        else:
            model = make_parzen(k).fit(X_train[indices], y_train[indices])
            row.update(
                {
                    "status": "ok",
                    **evaluate_classifier(
                        model, X_test, y_test, repeats=TIMING_REPEATS
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    output_dir: str | Path = OUTPUT_DIR,
) -> None:
    # 1. Загрузка из кэша KaggleHub и разделение на train/test.
    print("1. Подготовка данных", flush=True)
    data = DataProcessor(
        dataset_id, excluded_columns=EXCLUDED_COLUMNS, random_state=RANDOM_STATE
    )
    data.prepare(train_size=TRAIN_FRACTION)
    X_train, y_train, _ = data.get_split("train")
    X_test, y_test, _ = data.get_split("test")
    visualizer = Visualizer(data.feature_names, output_dir=output_dir)
    print(
        f"Полные выборки: train = {len(y_train)}, "
        f"test = {len(y_test)}, признаков = {X_train.shape[1]}",
        flush=True,
    )

    # 2. Реализация переменного окна с гауссовым ядром находится в make_parzen.
    print("2. Классификатор Парзена: переменное окно, гауссово ядро", flush=True)

    # 3. LOO использует только обучающие данные, без заранее рассчитанного масштаба.
    print("3. Подбор k по LOO", flush=True)
    X_loo, y_loo, _ = data.sample("train", LOO_SAMPLE_SIZE, standardized=False)
    k_parzen, k_sklearn, risk = select_k_loo(X_loo, y_loo, K_VALUES)
    risk["n_train"] = len(y_loo)
    print(f"Выбрано k: Парзен = {k_parzen}, sklearn KNN = {k_sklearn}")

    # 4. График риска для всех проверенных k.
    print("4. График эмпирического риска", flush=True)
    visualizer.plot_risk(risk)
    tables = {"loo_risk": risk}

    # 5. Сравнение моделей по всем признакам и отдельные иллюстрации в 2D.
    print("5. Сравнение со sklearn KNN", flush=True)
    reference_results = compare_with_sklearn(
        X_train, y_train, X_test, y_test, k_parzen=k_parzen, k_sklearn=k_sklearn
    )
    print(reference_results.to_string(index=False))
    tables["sklearn_comparison"] = reference_results
    visualizer.plot_comparison(reference_results, output_name="sklearn_comparison.png")

    def predict_parzen(X_fit, y_fit, queries):
        return make_parzen(k_parzen).fit(X_fit, y_fit).predict(queries)

    def predict_sklearn(X_fit, y_fit, queries):
        return (
            KNeighborsClassifier(n_neighbors=k_sklearn)
            .fit(X_fit, y_fit)
            .predict(queries)
        )

    X_plot, y_plot, _ = data.sample("train", PROJECTION_TRAIN_SIZE)
    X_plot_test, y_plot_test, _ = data.sample("test", PROJECTION_TEST_SIZE)
    visualizer.plot_classification(
        X_plot,
        y_plot,
        X_plot_test,
        y_plot_test,
        predictors={
            f"Парзен (k={k_parzen})": predict_parzen,
            f"sklearn KNN (k={k_sklearn})": predict_sklearn,
        },
    )

    # 6. Отбор по CCV для 1NN; control_size не является k классификатора Парзена.
    print("6. Отбор эталонов", flush=True)
    X_selection, y_selection, _ = data.sample("train", SELECTION_TRAIN_SIZE)
    X_selection_test, y_selection_test, _ = data.sample("test", SELECTION_TEST_SIZE)
    print(
        f"Подвыборки для отбора и его оценки: train = {len(y_selection)}, "
        f"test = {len(y_selection_test)}",
        flush=True,
    )
    selector = PrototypeSelector(
        X_selection, y_selection, control_size=CCV_CONTROL_SIZE
    )
    addition_indices, addition_history = selector.greedy_addition()
    tolerance, tolerance_results = select_removal_tolerance(
        selector, REMOVAL_TOLERANCES
    )
    removal_indices, removal_history = selector.greedy_removal(tolerance=tolerance)
    tables["removal_tolerances"] = tolerance_results
    print(f"Допуск удаления по CCV: {tolerance:g}")

    # 7. Качество выбранного Парзена; в 2D используются только две координаты.
    print("7. Визуализация отбора эталонов", flush=True)
    classifier_name = f"Парзен (k={k_parzen})"
    for operation, title, indices, history in (
        ("addition", "Жадное добавление", addition_indices, addition_history),
        ("removal", "Жадное удаление", removal_indices, removal_history),
    ):
        accuracy_history = evaluate_selection_history(
            X_selection,
            y_selection,
            X_selection_test,
            y_selection_test,
            history,
            k=k_parzen,
        )
        tables[f"{operation}_history"] = pd.DataFrame(
            {
                "step": [entry["step"] for entry in history],
                "n_prototypes": [entry["n_prototypes"] for entry in history],
                "ccv_1nn": [entry["ccv"] for entry in history],
                "parzen_k": k_parzen,
                "n_test": len(y_selection_test),
                "parzen_accuracy": accuracy_history,
            }
        )
        visualizer.plot_selection_history(
            history,
            title=title,
            test_errors=1 - accuracy_history,
            mark_ccv_minimum=operation == "removal",
            classifier_name=classifier_name,
            output_name=f"{operation}_history.png",
        )
        visualizer.plot_prototypes(
            X_selection,
            y_selection,
            X_selection_test,
            y_selection_test,
            indices,
            title=title,
            output_prefix=operation,
            predictor=predict_parzen,
            classifier_name=classifier_name,
            minimum_reference_size=k_parzen + 1,
        )
        gifs = visualizer.animate_selection(
            X_selection,
            y_selection,
            X_selection_test,
            y_selection_test,
            history,
            operation=operation,
            accuracy_all_features=accuracy_history,
            output_prefix=operation,
            predictor=predict_parzen,
            classifier_name=classifier_name,
            minimum_reference_size=k_parzen + 1,
            max_steps=SELECTION_GIF_MAX_STEPS,
        )
        for gif in gifs:
            visualizer.save_fast_gif(gif)

    # 8. Один выбранный Парзен, один test, три разных опорных набора.
    print("8. Сравнение Парзена с отбором и без отбора", flush=True)
    prototype_results = compare_support_sets(
        X_selection,
        y_selection,
        X_selection_test,
        y_selection_test,
        k=k_parzen,
        subsets={
            "Без отбора": np.arange(len(X_selection)),
            "Жадное добавление": addition_indices,
            "Жадное удаление": removal_indices,
        },
    )
    tables["parzen_comparison"] = prototype_results
    visualizer.plot_comparison(prototype_results)
    print(prototype_results.to_string(index=False))

    # 9. Сохранение численных результатов для отчёта.
    print("9. Сохранение таблиц для отчёта", flush=True)
    for name, table in tables.items():
        table.to_csv(visualizer.output_dir / f"{name}.csv", index=False)
    print(f"Графики, GIF и таблицы сохранены в {visualizer.output_dir.resolve()}")


if __name__ == "__main__":
    main()
