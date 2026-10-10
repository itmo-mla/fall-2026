import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from classifier import (
    correlation_init,
    fit_multistart,
    fit_sgd,
    fit_steepest,
    margin,
    mean_loss,
    predict,
    predict_scores,
)
from utils import (
    IMAGE_DIR,
    RESULTS_DIR,
    load_and_preprocess_data,
    metrics_for_custom,
    metrics_for_reference,
    save_margin_plot,
    save_metrics_plot,
    save_multistart_plot,
    save_q_plot,
    save_roc_plot,
    save_train_loss_plot,
)

RANDOM_STATE = 42
N_ITER = 5000
LEARNING_RATE = 0.001
GAMMA = 0.9
LAMBDA_Q = 0.01
TAU = 0.001
N_STARTS = 10


def fit_reference(X_train, y_train):
    """Библиотечный эталон с той же квадратичной целью и L2-штрафом."""
    reference = SGDClassifier(
        loss="squared_error",
        penalty="l2",
        # sklearn использует половину квадрата ошибки: alpha = tau / 2.
        alpha=TAU / 2.0,
        # Свободный член отключён, как и в собственной модели.
        fit_intercept=False,
        # optimal с малым alpha даёт слишком большой начальный шаг.
        learning_rate="invscaling",
        eta0=2.0 * LEARNING_RATE,
        power_t=0.5,
        # Здесь max_iter — эпохи; N_ITER у собственной модели — объекты.
        max_iter=N_ITER,
        tol=1e-6,
        random_state=RANDOM_STATE,
    )
    reference.fit(X_train, y_train)
    return reference


def main():
    # 1. Загрузка данных (Шаг 1)
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
        test_size=0.20, random_state=RANDOM_STATE
    )

    print("Признаков:", len(feature_names), "(без свободного члена)")
    print("Метки: -1 = malignant, +1 = benign; precision/recall/F1 для benign")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"Класс {int(cls):+d}: {count} объектов ({count / len(y_train):.1%})")

    # Инициализация через корреляцию (Шаг 9.i)
    w_corr = correlation_init(X_train, y_train)

    # Эксперимент 1: Корреляция + random sampling (Шаг 9.i, 9.iii)
    print("\n[1/5] Обучение: Correlation + Random...")
    corr_random = fit_sgd(
        X_train, y_train,
        w_init=w_corr,
        n_iter=N_ITER,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        lambda_q=LAMBDA_Q,
        tau=TAU,
        sampling="random",
        random_state=RANDOM_STATE,
    )

    # Эксперимент 2: Random init + Multistart (Шаг 9.ii)
    print("[2/5] Обучение: Random + Multistart...")
    multistart, starts = fit_multistart(
        X_train, y_train,
        n_starts=N_STARTS,
        random_state=RANDOM_STATE,
        n_iter=N_ITER,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        lambda_q=LAMBDA_Q,
        tau=TAU,
    )

    # Эксперимент 3: Корреляция + sampling по |margin| (Шаг 8, 9.iii)
    print("[3/5] Обучение: Correlation + |Margin| sampling...")
    corr_margin = fit_sgd(
        X_train, y_train,
        w_init=w_corr,
        n_iter=N_ITER,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        lambda_q=LAMBDA_Q,
        tau=TAU,
        sampling="margin",
        random_state=RANDOM_STATE,
    )

    # Эксперимент 4: Скорейший градиентный спуск (Шаг 7)
    print("[4/5] Обучение: Скорейший спуск (Steepest)...")
    steepest = fit_steepest(
        X_train, y_train,
        w_init=w_corr,
        n_iter=N_ITER,
        lambda_q=LAMBDA_Q,
        random_state=RANDOM_STATE,
    )

    # Эксперимент 5: Эталонная реализация из sklearn (Шаг 11)
    print("[5/5] Обучение: sklearn SGDClassifier baseline...")
    reference = fit_reference(X_train, y_train)

    # Шаг 10: Расчёт метрик
    custom = {
        "Correlation + random": corr_random,
        "Random + multistart": multistart,
        "Correlation + |margin|": corr_margin,
        "Steepest gradient": steepest,
    }

    rows = [
        metrics_for_custom(
            name, X_train, y_train, X_test, y_test, result["w"],
            mean_loss, predict, predict_scores
        )
        for name, result in custom.items()
    ]
    rows.append(
        metrics_for_reference("sklearn SGDClassifier", reference, X_train, y_train, X_test, y_test)
    )
    results = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК (Шаги 10 и 11):")
    print("=" * 60)
    print(results.round(4).to_string(index=False))

    print("\nТаблица стартов Multistart:")
    print(starts.round(5).to_string(index=False))

    # Сохранение результатов в CSV
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    starts.to_csv(RESULTS_DIR / "multistart_results.csv", index=False)

    # Построение и сохранение всех графиков в lab1/images
    save_q_plot({name: res["q_history"] for name, res in custom.items()})
    save_train_loss_plot({name: res["train_loss_history"] for name, res in custom.items()})
    save_metrics_plot(results)

    curves = {name: (y_test, predict_scores(X_test, res["w"])) for name, res in custom.items()}
    curves["sklearn SGDClassifier"] = (y_test, reference.decision_function(X_test))
    save_roc_plot(curves)

    save_margin_plot(X_train, y_train, corr_random["w"], margin)
    save_multistart_plot(starts)

    print(f"\n[+] Все графики сохранены в: {IMAGE_DIR}")
    print(f"[+] CSV таблицы сохранены в: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
