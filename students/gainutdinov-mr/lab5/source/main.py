"""Эксперименты лабораторной работы № 5. Логистическая регрессия."""

from pathlib import Path
from time import perf_counter

import numpy as np
from data import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from visualization import plot_results

OUTPUT_DIR = Path(__file__).with_name("figures")
RANDOM_STATE = 42


def sigmoid(x):
    # return np.exp(1/(1+np.exp(-x)))
    # Сигмоида без переполнения exp при больших отрицательных x.
    return np.exp(-np.logaddexp(0.0, -x))


def margin(X, y, w):
    return (X @ w) * y


def logistic_loss(y, scores):
    """
    Средняя логистическая потеря для меток -1/+1 и линейных оценок.
    # logaddexp(a, b) устойчиво вычисляет log(exp(a) + exp(b)).
    # При a=0, b=-M получаем log(1 + exp(-M)) без переполнения.
    """
    return np.mean(np.logaddexp(0.0, -y * scores))


def irls(F, y, tol=1e-5, max_iter=1000, *, history=None):
    """
    F - матрица (l, n), y - одномерный вектор меток (l,).

    Свободный член задаётся столбцом единиц в F - w[0].
    Возвращает коэффициенты w и число выполненных итераций.
    Если передан список history, добавляет в него историю сходимости.
    """
    # Начальное приближение - обычный МНК.
    w = np.linalg.pinv(F) @ y
    sigmas = sigmoid(margin(F, y, w))  # (l,)
    #
    # Для построения графиков того, как сходится наш метод, будем хранить историю
    if history is not None:
        history.append(
            {
                "iteration": 0,
                "train_loss": logistic_loss(y, F @ w),
                "probability_change": None,
            }
        )

    for iteration in range(1, max_iter + 1):
        # Ограничиваем вероятности только для вычисления весов и y_approx,
        # чтобы избежать деления на ноль при насыщении сигмоиды.
        safe_sigmas = np.clip(sigmas, 1e-12, 1 - 1e-12)
        gigachads = np.sqrt((1 - safe_sigmas) * safe_sigmas)  # gigachads, (l,)
        moggers = np.sqrt((1 - safe_sigmas) / safe_sigmas)  # (l,)
        # Умножение каждой строки F на свой gamma без создания diag(l, l).
        F_approx = np.diag(gigachads) @ F # Есть бродкастинг через gigachads[:, None] * F
        y_approx = y * moggers

        # МНК даёт приращение коэффициентов - направление Ньютона.
        delta_w = np.linalg.pinv(F_approx) @ y_approx
        # Полный шаг Ньютона
        # в общем случае может потребоваться уменьшение.
        h_t = 2 # Подобрал гиперпараметр
        w += h_t * delta_w

        sigmas_new = sigmoid(margin(F, y, w))
        probability_change = np.max(np.abs(sigmas_new - sigmas))

        if history is not None:
            history.append(
                {
                    "iteration": iteration,
                    "train_loss": logistic_loss(y, F @ w),
                    "probability_change": probability_change,
                }
            )

        if probability_change <= tol:
            break
        sigmas = sigmas_new

    return w, iteration


def compare_models(F_train, F_test, y_train, y_test, tol=1e-5):
    """Сравним наши модели"""
    history = []
    start = perf_counter()
    w, iterations = irls(F_train, y_train, tol=tol, history=history)
    irls_seconds = perf_counter() - start

    # Та же задача. Без регуляризации, свободный член уже входит в F.
    reference = LogisticRegression(
        C=np.inf,
        fit_intercept=False,
        solver="newton-cholesky",
        tol=1e-8,
        max_iter=1000,
    )
    start = perf_counter()
    reference.fit(F_train, y_train)
    reference_seconds = perf_counter() - start
    # Заведем для метрик отдельные хешмапы
    metrics = {}
    scores = {}
    models = (
        ("IRLS", w, iterations, irls_seconds),
        (
            "sklearn", # name
            reference.coef_[0], # weights
            int(reference.n_iter_[0]), # n_iter
            reference_seconds, # fit_seconds
        ),
    )
    for name, weights, n_iter, fit_seconds in models:
        scores[name] = F_test @ weights
        y_pred = np.where(scores[name] >= 0, 1, -1)
        metrics[name] = {
            "accuracy": np.mean(y_pred == y_test),
            "train_loss": logistic_loss(y_train, F_train @ weights),
            "test_loss": logistic_loss(y_test, scores[name]),
            # Используем линейные оценки: сигмоида может округлиться до 0/1
            # и потерять порядок объектов, который нужен для ROC AUC.
            "roc_auc": roc_auc_score(y_test, scores[name]),
            "fit_seconds": fit_seconds,
            "iterations": n_iter,
        }
        print(f"\n{name}")
        for metric, value in metrics[name].items():
            print(f"  {metric}: {value:.6g}")

    agreement = np.mean((scores["IRLS"] >= 0) == (scores["sklearn"] >= 0))
    print(f"Совпадение классов на тесте: {agreement:.2%}")
    return history, scores, metrics


def main():
    # 1. выбрать датасет для логистической регрессии
    X_train, X_test, y_train, y_test = load_data(
        test_size=0.3, random_state=RANDOM_STATE
    )
    # Столбец единиц соответствует свободному коэффициенту w[0].
    F_train = np.column_stack((np.ones(len(X_train)), X_train))
    F_test = np.column_stack((np.ones(len(X_test)), X_test))
    tol = 1e-5
    # 2. реализовать логистическую регрессию с помощью метода Ньютона-Рафсона и IRLS
    w, iterations = irls(F_train, y_train, tol=tol)
    num = 113
    print(f"{num+1}-й элемент = {np.sign((F_test @ w)[num])}\ntrue_label={y_test[num]}")
    # 3. показать эквивалентность с эталонной реализацией
    history, scores, metrics = compare_models(F_train, F_test, y_train, y_test, tol)
    plot_results(history, y_test, scores, metrics, OUTPUT_DIR, tol)
    print(f"Графики: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
