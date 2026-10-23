"""Собственная реализация алгоритма: SVM."""

import numpy as np

# TODO: реализовать алгоритм согласно tasks/task-03/README.md.
# По сути что мы делам в SVM
# Решаем двойственную задачу для lambda (та что на слайде)
# Для этого нам нужен какой-то оптимизатор например из scipy.optimize
# Для этого нам нужна будет обучающая выборка ({x[i], y[i]} for i in 1..l ),
# коэффицент C (чем больше тем меньше регуляризации, чем ближе к 0, тем наоборот)
# Ядро, которое удовлетворяет условиям. (А лучше уже заранее синтезированное ядро)
#
# А еще по найденным lambda мы получается сможем ответить на вопрос,
# какие объекты стали опорными. И не опорные - выкинуть, так как там lambda=0
# Коэффицент C - гиперпараметр. Можно подобрать используя grid search.
# Т.е при разных C решаем задачу, и например с помощью LOO, выбираем лучший коэфф.
# Хотя по факту результат слабо зависит от C
# Ядро - K - тоже гиперпараметр.
#
# Крч, предлагаю создать класс, у которого будут поля
# Kernel
# lambdas
# X - объекты обучающей выборки
# y - их метки
# C - коэфф регуляризации
# kernel_matrix = квадратная матрица ядер где перебираются j = 1..n; i = 1..n
# X_support - Опорные объекты
# Методы
# fit
# predict
# ...
# Скрытые
# оптимизировать - найти решение двойственной задачи
# ядро K_1 .... K_N
# Показать опорные объекты// индексы опорных объектов// или если мы и так храним исходную выборку в классе, то просто сами объекты и соответствующие им таргеты
#
# Получить ширину полосы - 2 / ||W||
#
# ...
from data import load_data, standardize_data
from scipy import optimize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
y = 2 * y - 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_train, X_test, _, _ = standardize_data(X_train, X_test)


def linear_kernel(X):
    return X @ X.T


def polynomial_kernel(X, degree):
    return (linear_kernel(X) + 1) ** degree


def kernel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a @ b


def lazy_objective(lambdas):
    return -np.sum(lambdas) + 0.5 * np.sum(
        [
            [
                lambdas[i] * lambdas[j] * y[i] * y[j] * kernel(X[i], X[j])
                for j in range(len(X))
            ]
            for i in range(len(X))
        ]
    )


# Пусть alhpa = lambda [i] * y[i] = (l, 1); K = l x l
# как бы переписать lambdas[i] * lambdas[j] * y[i] * y[j] * kernel(X[i], X[j]) через alpha?
# можно попробовать так, чтобы было (1 x l) x (l x l) x (l x 1) чтобы получился скаляр
# тогда это типо alpha.T @ K @ alpha, но для numpy эквивалентно alpha @ K @ alpha
K = polynomial_kernel(X_train, 2)


def numpy_objective(lambdas):
    alpha = lambdas * y_train
    return -np.sum(lambdas) + 0.5 * (alpha @ K @ alpha)


def gradient(lambdas):
    alpha = lambdas * y_train
    return y_train * (K @ alpha) - 1


def objective_and_gradient(lambdas):
    alpha = lambdas * y_train
    K_alpha = K @ alpha

    objective = -lambdas.sum() + 0.5 * (alpha @ K_alpha)
    gradient = y_train * K_alpha - 1

    return objective, gradient


if __name__ == "__main__":
    C = 1.0

    initial_assumption = np.zeros(len(X_train))

    bound = optimize.Bounds(0, C)
    constrains = optimize.LinearConstraint(y_train, lb=0, ub=0)

    # res = optimize.minimize(
    #     lazy_objective,
    #     initial_assumption,
    #     bounds=bound,
    #     constraints=constrains,
    #     options={"maxiter": 1000},
    # )
    # res = optimize.minimize(
    #     numpy_objective,
    #     initial_assumption,
    #     jac=gradient,
    #     method="SLSQP",
    #     bounds=bound,
    #     constraints=constrains,
    #     options={
    #         "maxiter": 1000,
    #         "ftol": 1e-8,
    #     },
    # )
    res = optimize.minimize(
        objective_and_gradient,
        initial_assumption,
        jac=True,
        method="SLSQP",
        bounds=bound,
        constraints=constrains,
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
        },
    )
    print(res.success)
    print(res.message)
    if not res.success:
        raise RuntimeError("Optimization failed")

    lambdas = res.x

    # print(lambdas)
    # X = l x n
    # y = l x 1
    # lambdas = l x 1
    # weights = n x 1
    # weigts = X.T @ (lambdas o y) = (n x l @ l x 1) = (n x 1)
    # weights = X.T @ (lambdas * y)
    # weights = X_train.T @ (lambdas * y_train)
    eps = 1e-5

    # margin_mask = (lambdas > eps) & (lambdas < C - eps) # Возьмем и посчитаем средний w0 если бы мы брали для него только те объекты, которые опорные-граничные
    # т.е. M[i] = 1 для них.
    # w0 = np.mean(X_train[margin_mask] @ weights - y_train[margin_mask])

    for lambda_idx in range(len(lambdas)):
        if 0 < lambdas[lambda_idx] < C:
            idx = lambda_idx
            break

    w0 = (
        np.sum(
            [
                lambdas[j] * y[j] * kernel(X_train[idx], X_train[j])
                for j in range(len(X_train))
            ]
        )
        - y[idx]
    )

    def predict(X):
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        scores = np.asarray(
            [
                np.sum(
                    [
                        lambdas[i] * y[i] * kernel(sample, X_train[i])
                        for i in range(len(X_train))
                    ]
                )
                for sample in X
            ]
        )
        return np.where(scores >= 0, 1, -1)

    train_predictions = predict(X_train)
    test_predictions = predict(X_test)
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    print("Support vectors:", np.sum(lambdas > eps))
    print("w0:", w0)
    print("Prediction:", predict(X_test[2]))
    print("True label:", y_test[2])
