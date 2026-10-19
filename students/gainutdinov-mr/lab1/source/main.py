"""
В рамках лабораторной работы предстоит реализовать линейный классификатор.
И обучить его методом стохастического градиентного спуска с инерцией
с L2 регуляризацией и квадратичной функцией потерь.
"""

# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?&tags=13304-Clustering);
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEARNING_RATE = 0.0003
FORGETTING_RATE = 0.01
MOMENTUM = 0.9
REGULARIZATION = 0.1





# print(f'Path to dataset file: {path_to_zoo}\n')
# print(pd.read_csv(train_path,sep=','))

# print()

# df_train = pd.read_csv(train_path)
# df_test = pd.read_csv(test_path)
#
# print(f"ANY NULL??: {df_train.isna().any().any()}")
# print(df_train.columns[df_train.isna().any()].tolist())
# df_train = df_train.dropna()
# print(df_train.head())
# print(df_train.shape)
#
#
# print(df_train.describe(include="all"))
# print(f"{df_train.info()}\n\n")
#
#
# print(f"ANY NULL??: {df_test.isna().any().any()}")
# print(df_test.columns[df_test.isna().any()].tolist())
#
# df_test = df_test.dropna()
# print(df_test.head())
# print(df_test.shape)
#
#
# print(df_test.describe(include="all"))
# print(f"{df_test.info()}\n\n")

# df_test.drop(["Unnamed: 0", "id"], axis=1, inplace=True)


# Объект - пассажир
# Классы - удовлетворен/ нейтрально или неудовлетворен полетом

# Цель - предсказать класс удовлетворения полетом (satisfied/ neutral or dissatisfied) на основе признаков пассажира.

# Без итогового класса - у нас 24 признка,

# Первый столбец - нам не нужен, так как он лишь обозначает номер записи, и к пассажиру никакого отношения не имеет.
# id - возможно стоит удалить, но ради эксперимента оставим.

# Признаки
# Бинарные: Gender, Customer Type, Type of Travel
# Номинальные:
# Порядковые: Class (Eco = 0, EcoPlus = 1, Business = 2), это как tier подписки
# Количественные:   Age, Flight Distance,
#
#                   Infliflight wifi service, Departure/Arrival time convenient,
#                   Ease of Online booking, Gate location, Food and drink,
#                   Online boarding, Seat comfort, Inflight entertainment,
#                   On-board service, Leg room service, Baggage handling,
#                   Checkin service, Inflight service, Cleanliness,
#
#                   Departure Delay in Minutes, Arrival Delay in Minutes


# Для Arrival Delay in Minutes было выявленно 25893 non-null вхождений - это в тесте,
# Получается, что 25976-25893 = 83 пассажира не указали время задержки прибытия.


def prepare_data(data_path, DROP_NA=True):
    # TODO: Вернуть предобработанные данные в виде pandas df
    # Если планируется применять эту функцию к тестовой и трейн данным, то лучше дополнительно проверить на утечку
    df = pd.read_csv(data_path)
    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)
    if DROP_NA:
        df.dropna(inplace=True)

    # print(df.head())

    # gender_values = [x for x in df["Gender"].unique()]
    # customer_type_values = [x for x in df["Customer Type"].unique()]
    # type_of_travel_value = [x for x in df["Type of Travel"].unique()]
    # # class_values = [x for x in df["Class"].unique()]
    # satisfaction_values = [x for x in df["satisfaction"].unique()]

    # print(f"Gender values = {gender_values}")
    # print(f"Customer Type values = {customer_type_values}")
    # print(f"Type of Travel values = {type_of_travel_value}")
    # # print(f"Class values = {class_values}")
    # print(f"satisfaction values = {satisfaction_values}")

    rule_satisfied = {
        "satisfied": 1,
        "neutral or dissatisfied": -1,
    }  # 1 and -1 потому что у нас бинарная классификация
    rule_gender = {"Male": 0, "Female": 1}
    rule_customer_type = {"Loyal Customer": 0, "disloyal Customer": 1}
    rule_type_of_travel = {"Personal Travel": 0, "Business travel": 1}
    rule_class = {"Eco": 0, "Eco Plus": 1, "Business": 2}  # Так как их >2, пропишу явно

    df.replace({"satisfaction": rule_satisfied}, inplace=True)
    df.replace({"Gender": rule_gender}, inplace=True)
    df.replace({"Customer Type": rule_customer_type}, inplace=True)
    df.replace({"Type of Travel": rule_type_of_travel}, inplace=True)
    df.replace({"Class": rule_class}, inplace=True)
    # TODO: Добавить свободный член для w0

    return df


def standardize_data(X_train, X_test):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Чтобы не делить на ноль, где например постоянные признаки.
    train_std = np.where(train_std == 0, 1.0, train_std)

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    return X_train_scaled, X_test_scaled, train_mean, train_std


# print(X_train.columns)
# www = np.zeros_like(X_train.columns)
# print(www)
# print(len(www))
# print(len(X_train.columns))
# print(X_train.head())
# print("Train data:\n", X_train)
# print("Test data:\n", X_test)

# print("Train data:\n", y_train)
# print("Test data:\n", y_test)

# return (X_train, y_train), (X_test, y_test)
# 2. реализовать вычисление отступа объекта (визуализировать, проанализировать);

# Для визуализации используем matplotlib, по идее нам надо в 2D разместить точки,
# затем провести линию M_i(w) = g(x_i, w)y_i, используя
# разделяющей функции g(x,w) (=0 для бинарного классификатора)
#


def calculate_margin(X, w, y):
    # Написать функцию, которая будет вычислять отступ объекта.
    # g (x, w) = sign(Sum_{1}^{len(w)} w_j * f_j(x))), но для формулы функцию sign убирают.
    # где f_j(x) - j-итая фича объекста x, а w_j - подобранный (чудом) вес для нее.

    # Отступ для i-го объекта (x_i из множества X) = g(x_i, w) * y_i
    # Веса для фич что для одного объекта, что для другого из X будут одинаковые
    # Если писать циклом:
    # M = []
    # for i in range(len(X)):
    #    # M_i = g(...) * Y[i]
    #    #
    #    # x_i = X[i]
    #    # y_i = y[i]
    #    # summ = 0
    #    # for j in range(len(W)):
    #    #     summ += w[j] * x_i[j]  ## ==> w.T @ x
    #    # M.append(summ * y_i) ## ==> summ = w.T @ X.T ==> summ = X @ w
    # return M

    return (X @ w) * y


def calculate_margin_i(features, w, target):
    # w.T = R^1 x n; features = R^n x 1;
    return (
        w.T @ features
    ) * target  # Получается число - Margin для объекта с фичами X[i], таргетом y[i].


def plot_margin(margin, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.sort(margin), label="Отступы")
    ax.axhline(0, color="red", linestyle="--", label="Граница классификации")
    ax.axhline(1, color="green", linestyle=":", label="Минимум потери")
    ax.set(title=title, xlabel="Объекты, отсортированные по отступу", ylabel="M")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_margin(margin):
    # Разделим результат на 3 группы качества:
    # 1) (-inf; -eps) - отвратительно
    # 2) [-eps; eps] - было близко
    # 3) (eps; +inf) - хорошо
    eps = 0.5
    confident_miss = np.sum(margin < -eps)
    close = np.sum((margin >= -eps) & (margin <= eps))
    accepted = np.sum(margin > eps)
    n_wrong = np.sum(margin <= 0)
    print(
        f"Существенный промах: {confident_miss}\n"
        f"Почти: {close}\n"
        f"Приемлимый результат: {accepted}\n"
        f"\n"
        f"Доля правильных ответов: {(len(margin) - n_wrong) / len(margin)}\n"
        f"Средний отступ: {margin.mean()}"
    )


# 3. реализовать вычисление градиента функции потерь;
# Это понятно, градиент. (Можно как написать ручками, так и использовать numpy)


def loss_function(margin, func_name="FLD"):
    # По заданию - квадратичная функция потерь - FLD
    return 0.5 * (1 - margin) ** 2


def loss_gradient(sample, weights, target):

    # M[i] = y[i] * (w.T @ x[i])
    # Loss[i] = 1/2 * (1-M[i])**2
    # =>
    # grad(L[i]) = dL[i]/dM[i] * dM[i]/dW
    # (1-M[i])(-1) * y[i] * x[i][k] - частные производные по w[k]
    # Можно вынести (1-M[i])(-1) * y[i] за скобку, и тогда можно посчитать градиент
    # grad(L[i]) = ((M[i] - 1) * y[i]) * x[i]
    # x[i] - вектор длинной с размером фич
    margin = calculate_margin(sample, weights, target)
    return ((margin - 1) * target) * sample


# 4. реализовать рекуррентную оценку функционала качества;
# Q - от слова Quality
def update_quality(old_quality, current_loss):
    return FORGETTING_RATE * current_loss + (1 - FORGETTING_RATE) * old_quality
    # return new_q_value = FORGETTING_RATE * Loss (x_i) + (1 - FORGETTING_RATE) * old_q_value


def get_presentation_indices(X, y, weights, presentation, rng=None):
    rng = np.random.default_rng(rng)
    if presentation == "random":
        return rng.permutation(len(X))

    if presentation == "margin":
        margins = calculate_margin(X, weights, y)
        probabilities = 1 / (1 + np.abs(margins))
        probabilities /= probabilities.sum()
        # Выбор с возвращением: объекты у границы могут встретиться чаще.
        return rng.choice(len(X), size=len(X), replace=True, p=probabilities)

    raise ValueError("presentation это или 'random' или 'margin'")


# 5. реализовать метод стохастического градиентного спуска с инерцией;


def SGD_with_momentum(
    X,
    y,
    initial_weights,
    max_iterations,
    momentum=0.9,
    regularization=0.1,
    presentation="random",
    random_state=None,
):
    if len(X) == 0:
        raise ValueError("Обучающая выборка не должна быть пустой")
    if presentation not in ("random", "margin"):
        raise ValueError("presentation это или 'random' или 'margin'")

    rng = np.random.default_rng(random_state)
    weights = initial_weights.copy()

    subset_size = min(
        100, len(X)
    )  # Инициализируем qaulity по случайному подмножеству из 100 элементов, можно больше, можно меньше. Можно поиграться
    subset_indices = rng.choice(len(X), size=subset_size, replace=False)

    initial_margins = calculate_margin(X[subset_indices], weights, y[subset_indices])

    initial_l2_penalty, _ = l2_penalty_and_gradient(weights, regularization)

    quality = np.mean(loss_function(initial_margins)) + initial_l2_penalty

    quality_history = []  # - Чтобы потом показывать график обучения
    quality_history.append(quality)

    velocity = np.zeros_like(weights)

    # v = gamma * v + (1 - gamma) * grad(Loss(w,x[i])) # - velocity
    # w = w - LEARNING_RATE * v # weights

    iteration = 0
    while iteration < max_iterations:
        presentation_order = get_presentation_indices(
            X,
            y,
            weights,
            presentation,
            rng,
        )

        for index in presentation_order:
            sample = X[index]
            target = y[index]

            margin = calculate_margin(sample, weights, target)
            current_loss = loss_function(margin)
            gradient = loss_gradient(sample, weights, target)

            # L2 регуляризация
            l2_penalty, l2_gradient = l2_penalty_and_gradient(
                weights,
                regularization,
            )
            current_loss += l2_penalty
            gradient += l2_gradient  # l2_gradient[0] == 0, поэтому w0 не регуляризуется

            velocity = momentum * velocity + (1 - momentum) * gradient
            weights = weights - LEARNING_RATE * velocity
            quality = update_quality(quality, current_loss)
            quality_history.append(quality)

            iteration += 1
            if iteration >= max_iterations:
                break

    return weights, quality_history


# 6. реализовать L2 регуляризацию;
# AKA Tikhonov regularization or ridge regression
def l2_penalty_and_gradient(weights, regularization):
    # regularization - коэфф регуляризации
    #
    weights_regularized = weights.copy()
    weights_regularized[0] = (
        0  # Так как для w0 мы регуляризацию не делаем. # TODO: А может сделать?
    )

    penalty = regularization / 2 * np.sum(weights_regularized**2)
    gradient = regularization * weights_regularized

    return penalty, gradient
    # tau = reg_coeff
    # loss_modified = Loss(w, x[i]) + tau/2 * sum(w_j**2)
    # grad_loss_modified = grad_loss(w, x[i]) + tau * w
    # =>
    # w = w * (1 - LEARNING_RATE*tau) - LEARNING_RATE * grad_loss(w, x[i])
    # pass


# 7. реализовать скорейший градиентный спуск;
def optimal_step(sample):
    # Это работает только когда у нас квадратичная потеря - MSE
    squared_norm = sample @ sample

    if squared_norm == 0:
        return 0.0
    return 1.0 / squared_norm


def stochastic_steepest_gradient_descent(
    X,
    y,
    initial_weights,
    max_iterations,
):
    # TODO: все тоже самое, только LEARNING_RATE = optimal_step(sample)
    weights = initial_weights.copy()

    subset_size = min(100, len(X))
    subset_indices = np.random.choice(
        len(X),
        size=subset_size,
        replace=False,
    )

    initial_margins = calculate_margin(
        X[subset_indices],
        weights,
        y[subset_indices],
    )

    quality = np.mean(loss_function(initial_margins))
    quality_history = [quality]

    for iteration in range(max_iterations):
        index = np.random.randint(len(X))
        sample = X[index]
        target = y[index]

        margin = calculate_margin(
            sample,
            weights,
            target,
        )
        current_loss = loss_function(margin)

        gradient = loss_gradient(
            sample,
            weights,
            target,
        )

        learning_rate = optimal_step(sample)

        weights = (
            weights
            - learning_rate * gradient
        )

        quality = update_quality(
            quality,
            current_loss,
        )
        quality_history.append(quality)

    return weights, quality_history


def batch_steepest_gradient_descent(
    X,
    y,
    initial_weights,
    max_iterations,
    tolerance=1e-8,
):
    weights = initial_weights.copy()
    n_samples = len(X)

    # Для квадратичной функции гессиан постоянен,
    # поэтому достаточно вычислить его один раз.
    hessian = (X.T @ X) / n_samples

    initial_margins = calculate_margin(
        X,
        weights,
        y,
    )
    initial_quality = np.mean(
        loss_function(initial_margins)
    )

    quality_history = [initial_quality]
    step_history = []

    for iteration in range(max_iterations):
        predictions = X @ weights
        residuals = predictions - y

        gradient = (
            X.T @ residuals
        ) / n_samples

        gradient_squared_norm = gradient @ gradient

        # Если градиент практически нулевой,
        # мы уже находимся рядом с минимумом.
        if gradient_squared_norm <= tolerance**2:
            break

        hessian_times_gradient = hessian @ gradient
        denominator = gradient @ hessian_times_gradient

        # Защита от деления на ноль из-за
        # вырожденности или численных погрешностей.
        if denominator <= np.finfo(float).eps:
            break

        learning_rate = (
            gradient_squared_norm
            / denominator
        )

        weights = (
            weights
            - learning_rate * gradient
        )

        margins = calculate_margin(
            X,
            weights,
            y,
        )
        quality = np.mean(
            loss_function(margins)
        )

        quality_history.append(quality)
        step_history.append(learning_rate)

    return weights, quality_history, step_history

# 8. реализовать предъявление объектов по модулю отступа;
# То есть чаще брать объекты, на которых уверенность меньше:
# чем меньше |Mi |, тем больше вероятность взять объект;

# Это получается, надо сначала как-то прогнать модель, и получить результаты по каждому объекту
# Затем высчитать для каждого из них |Mi| (т.е. Отступ для i-го объекта)
# Как-то их ранжировать или придумать, как брать именно те объекты, на которых малейший отступ.
# Т.е. если |M_i| -> 0, то Вероятность взять i-й объект -> 1;


def init_weights(X, y):
    # Реализовать инициализацию весов через корреляцию
    # X - feature matrix; Т.е. из нее мы можем достать например столбец j-го признака
    # y - targets.
    # weights - вектор весов
    #
    #
    # Также, если Функция потерь квадратична, а признаки некоррелированы <f_k, f_j> = 0, j!=k,
    # то мы уже инициализировались итоговыми весами
    #
    y = y.to_numpy()
    weights = np.zeros(
        X.shape[1]
    )  # TODO: Сначала массив можно проинициализировать нулями, и он должен быть размером
    for j in range(len(weights)):
        f_j = X.iloc[:, j].to_numpy()  # Feature column - j
        weights[j] = (y @ f_j) / (f_j @ f_j)

    return weights


def init_weights_v2(X, y):
    # По сути, если правильно расписать линал в оригинальной функции, можно понять, что ее можно записать куда короче
    # используя уже встроенный функционал numpy
    X = np.asarray(X)
    y = np.asarray(y)

    return (X.T @ y) / np.sum(X**2, axis=0)


# 9. обучить линейный классификатор на выбранном датасете;
#    1. обучить с инициализацией весов через корреляцию;
#    2. обучить со случайной инициализацией весов через мультистарт;
#    3. обучить со случайным предъявлением и с п.8;


def init_weights_random(n_features, rng):
    # n_features включает столбец единиц для свободного коэффициента.
    if n_features < 1:
        raise ValueError("n_features должен быть положительным")
    bound = 1.0 / (2 * n_features)
    return rng.uniform(low=-bound, high=bound, size=n_features)


def calculate_regularized_quality(X, y, weights, regularization):
    # Точный функционал на всей переданной выборке, без сглаживания.
    margins = calculate_margin(X, weights, y)
    penalty, _ = l2_penalty_and_gradient(weights, regularization)
    return float(np.mean(loss_function(margins)) + penalty)


def train_with_multistart(
    X,
    y,
    n_starts,
    max_iterations,
    momentum=0.9,
    regularization=0.1,
    presentation="random",
    random_state=None,
):
    if n_starts < 1:
        raise ValueError("n_starts должен быть положительным")
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть положительным")

    rng = np.random.default_rng(random_state)
    best_quality = np.inf
    best_weights = None
    best_history = None
    results = []

    for start in range(n_starts):
        # Seed позволяет повторить и инициализацию, и обучение этого старта.
        seed = int(rng.integers(0, 2**32))
        start_rng = np.random.default_rng(seed)
        initial_weights = init_weights_random(X.shape[1], start_rng)
        trained_weights, history = SGD_with_momentum(
            X,
            y,
            initial_weights,
            max_iterations,
            momentum=momentum,
            regularization=regularization,
            presentation=presentation,
            random_state=start_rng,
        )

        quality = calculate_regularized_quality(
            X, y, trained_weights, regularization
        )
        # Разошедшийся запуск сохраняем для анализа, но не выбираем лучшим.
        is_finite = bool(np.isfinite(trained_weights).all() and np.isfinite(quality))
        results.append({
            "start": start + 1,
            "seed": seed,
            "initial_weights": initial_weights,
            "weights": trained_weights,
            "quality": quality,
            "is_finite": is_finite,
        })

        if is_finite and quality < best_quality:
            best_quality = quality
            best_weights = trained_weights.copy()
            best_history = history

    if best_weights is None:
        raise RuntimeError("Все старты разошлись: проверьте темп обучения и данные")

    return best_weights, best_history, results


# 10. оценить качество классификации;
def predict(X, weights):
    scores = X @ weights
    return np.where(scores >= 0, 1, -1)

def classification_metrics(y_true, y_pred):
    """Метрики для меток {-1, +1}; положительный класс — +1.

    Матрица ошибок: строки — истинный класс, столбцы — предсказанный.
    Порядок классов [-1, +1], поэтому матрица имеет вид [[TN, FP], [FN, TP]].
    При нулевом знаменателе precision, recall и F1 возвращаются как 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true и y_pred должны быть одномерными массивами")
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true и y_pred должны иметь одинаковую ненулевую длину")
    if not (np.isin(y_true, [-1, 1]).all() and np.isin(y_pred, [-1, 1]).all()):
        raise ValueError("Метки должны быть равны -1 или +1")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    accuracy = (tp + tn) / y_true.size
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def MeanSquareError(model, target):
    return np.mean((target - model) ** 2)


# 11. сравнить лучшую реализацию с эталонной; эталлонная - из готовых библиотек


# 12. подготовить отчет.
# TODO: Написать ридми

if __name__ == "__main__":
    path = Path(kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction"))
    train_path = path / "train.csv"
    test_path = path / "test.csv"

    train_data = prepare_data(train_path)
    test_data = prepare_data(test_path)
    X_train = train_data.drop(["satisfaction"], axis=1)
    y_train = train_data["satisfaction"]
    X_test = test_data.drop(["satisfaction"], axis=1)
    y_test = test_data["satisfaction"]
    X_train = X_train.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)

    X_train, X_test, feature_mean, feature_std = standardize_data(
        X_train,
        X_test,
    )
    # Первый столбец равен единице: его коэффициент и есть свободный член w0.
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    initial_weights = init_weights_v2(X_train, y_train)
    epochs = 10
    max_iterations = epochs * len(X_train)
    random_state = 42
    multistart_runs = 5

    # В каждой строке — модель, вычисленная из одних и тех же train-признаков.
    models = []

    correlation_random_weights, correlation_random_history = SGD_with_momentum(
        X_train, y_train, initial_weights, max_iterations,
        momentum=MOMENTUM, regularization=REGULARIZATION,
        presentation="random", random_state=random_state,
    )
    models.append(("Корреляция + случайный порядок", correlation_random_weights,
                   correlation_random_history, REGULARIZATION))

    multistart_random_weights, multistart_random_history, random_starts = train_with_multistart(
        X_train, y_train, n_starts=multistart_runs, max_iterations=max_iterations,
        momentum=MOMENTUM, regularization=REGULARIZATION,
        presentation="random", random_state=random_state,
    )
    models.append(("Мультистарт + случайный порядок", multistart_random_weights,
                   multistart_random_history, REGULARIZATION))

    correlation_margin_weights, correlation_margin_history = SGD_with_momentum(
        X_train, y_train, initial_weights, max_iterations,
        momentum=MOMENTUM, regularization=REGULARIZATION,
        presentation="margin", random_state=random_state,
    )
    models.append(("Корреляция + выбор по |M|", correlation_margin_weights,
                   correlation_margin_history, REGULARIZATION))

    multistart_margin_weights, multistart_margin_history, margin_starts = train_with_multistart(
        X_train, y_train, n_starts=multistart_runs, max_iterations=max_iterations,
        momentum=MOMENTUM, regularization=REGULARIZATION,
        presentation="margin", random_state=random_state,
    )
    models.append(("Мультистарт + выбор по |M|", multistart_margin_weights,
                   multistart_margin_history, REGULARIZATION))

    # Формула оптимального шага для одного объекта не учитывает L2 и momentum.
    np.random.seed(random_state)
    stochastic_weights, stochastic_history = stochastic_steepest_gradient_descent(
        X_train, y_train, initial_weights, max_iterations,
    )
    models.append(("Стохастический скорейший спуск", stochastic_weights,
                   stochastic_history, 0.0))

    batch_weights, batch_history, batch_steps = batch_steepest_gradient_descent(
        X_train, y_train, initial_weights, max_iterations=10000,
    )
    models.append(("Пакетный скорейший спуск", batch_weights,
                   batch_history, 0.0))

    from sklearn.linear_model import RidgeClassifier

    ridge_l2 = RidgeClassifier(alpha=len(X_train) * REGULARIZATION)
    ridge_l2.fit(X_train[:, 1:], y_train)
    ridge_l2_weights = np.r_[ridge_l2.intercept_, ridge_l2.coef_.ravel()]
    models.append(("sklearn RidgeClassifier + L2", ridge_l2_weights, None,
                   REGULARIZATION))

    ridge_no_l2 = RidgeClassifier(alpha=0.0)
    ridge_no_l2.fit(X_train[:, 1:], y_train)
    ridge_no_l2_weights = np.r_[ridge_no_l2.intercept_, ridge_no_l2.coef_.ravel()]
    models.append(("sklearn RidgeClassifier без L2", ridge_no_l2_weights,
                   None, 0.0))

    print(f"Train: {len(X_train)}, test: {len(X_test)}, признаков с w0: {X_train.shape[1]}")
    print(f"Шаг SGD: {LEARNING_RATE}, momentum: {MOMENTUM}, L2: {REGULARIZATION}")
    print(f"SGD: {epochs} прохода, мультистарт: {multistart_runs} запусков, seed: {random_state}")
    for mode, runs in (("случайный порядок", random_starts),
                       ("выбор по |M|", margin_starts)):
        best_start = min((run for run in runs if run["is_finite"]),
                         key=lambda run: run["quality"])
        print(f"Мультистарт ({mode}): выбран старт {best_start['start']} "
              f"из {multistart_runs}, train Q_reg={best_start['quality']:.6f}")

    output_dir = Path(__file__).parent.parent / "results" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    metrics_for_plot = []
    for model_number, (name, weights, history, regularization) in enumerate(models, start=1):
        train_metrics = classification_metrics(y_train, predict(X_train, weights))
        test_metrics = classification_metrics(y_test, predict(X_test, weights))
        train_loss = np.mean(loss_function(calculate_margin(X_train, weights, y_train)))
        test_margins = calculate_margin(X_test, weights, y_test)
        test_loss = np.mean(loss_function(test_margins))
        train_objective = calculate_regularized_quality(
            X_train, y_train, weights, regularization,
        )
        print(f"\n{name}")
        print(f"  Train: accuracy={train_metrics['accuracy']:.4f}, "
              f"precision={train_metrics['precision']:.4f}, "
              f"recall={train_metrics['recall']:.4f}, F1={train_metrics['f1']:.4f}, "
              f"loss={train_loss:.4f}, Q_reg={train_objective:.4f}")
        print(f"  Test:  accuracy={test_metrics['accuracy']:.4f}, "
              f"precision={test_metrics['precision']:.4f}, "
              f"recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, "
              f"loss={test_loss:.4f}")
        print(f"  TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
              f"FP={test_metrics['fp']}, FN={test_metrics['fn']}")
        print(f"  Матрица ошибок [[TN, FP], [FN, TP]]: "
              f"{test_metrics['confusion_matrix'].tolist()}")
        print("  Анализ отступов на test:")
        analyze_margin(test_margins)
        margin_plot = output_dir / f"margin_{model_number:02d}.png"
        plot_margin(test_margins, name, margin_plot)
        print(f"  График отступов: {margin_plot}")
        metrics_for_plot.append(test_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for name, _, history, _ in models[:4]:
        indices = np.linspace(0, len(history) - 1, min(2000, len(history)), dtype=int)
        axes[0].plot(indices / len(X_train), np.asarray(history)[indices],
                     label=name, linewidth=0.8)
    axes[0].set(title="SGD: рекуррентная оценка Q", xlabel="Проходы", ylabel="Q")
    axes[0].legend(fontsize=7)

    indices = np.linspace(0, len(stochastic_history) - 1,
                          min(2000, len(stochastic_history)), dtype=int)
    axes[1].plot(indices / len(X_train), np.asarray(stochastic_history)[indices],
                 color="tab:orange", linewidth=0.8)
    axes[1].set(title="Стохастический скорейший: рекуррентный Q",
                xlabel="Эквиваленты проходов", ylabel="Q")

    axes[2].plot(np.arange(len(batch_history)), batch_history, label="Batch")
    axes[2].axhline(calculate_regularized_quality(X_train, y_train,
                       ridge_no_l2_weights, 0.0), linestyle="--", color="black",
                    label="Ridge без L2")
    axes[2].set(title="Пакетный спуск: точный train loss", xlabel="Итерации",
                ylabel="Loss")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    labels = [name for name, _, _, _ in models]
    x_positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_positions - width / 2,
           [m["accuracy"] for m in metrics_for_plot], width, label="Accuracy")
    ax.bar(x_positions + width / 2,
           [m["f1"] for m in metrics_for_plot], width, label="F1 (+1)")
    ax.set(ylim=(0, 1), ylabel="Значение метрики", title="Качество на test")
    ax.set_xticks(x_positions, labels, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "test_metrics.png", dpi=160)
    plt.close(fig)
    print(f"\nГрафики сохранены в: {output_dir}")
