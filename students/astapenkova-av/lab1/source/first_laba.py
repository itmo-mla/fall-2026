import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

data = pd.read_csv("/Users/arlinrus/Desktop/fall-2026/students/astapenkova-av/lab1/source/diabetes_risk.csv", sep=',')

h = 1e-5 # шаг
lambda_ = 0.001 # какая доля новой ошибки учитывается в новом Q
gamma = 0.9 # насколько сильно учитывается предыдущая инерция
q_sz = 100 # выбираются примрено 100 случайных обектов
eps = 1e-6 #порог чувствительности
tau = 1e-3 #l2 регуляризация

# print("Размер датасета:", data.shape)

# print("\nНазвания колонок:")
# print(data.columns.tolist())

# print("\nИнфо:")
# print(data.info())

# print("\ПРопущенные значения:")
# print(data.isnull().sum())

# # дубликаты
# print(data.duplicated().sum())

target_column = 'diabetes_risk' # таргетированная колонка(риск диабета)

X = data.drop(columns=[target_column, "patient_id"])
y = data["diabetes_risk"].map({
    "Low": -1,
    "Moderate": 1,
    "High": 1
})

print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42,stratify=y)

X_train = X_train.copy()
X_test = X_test.copy()

mode_columns = ["alcohol_consumption", "smoking_status", "income_bracket"]
for column in mode_columns:
    train_mode = X_train[column].mode(dropna=True)
    fill_value = train_mode.iloc[0]
    X_train[column] = X_train[column].fillna(fill_value)
    X_test[column] = X_test[column].fillna(fill_value)
categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# print(categorical_columns)

X_train = pd.get_dummies(X_train, columns=categorical_columns,drop_first=True)

X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

n_features = X_train.shape[1]
print(n_features)

w = np.zeros(n_features)

# Реализовать вычисление отступа объекта (визуализировать, проанализировать)
def margin_classifier(X, y, w):
  X = np.asarray(X)
  y = np.asarray(y)
  return y * (X @ w) 
margins = margin_classifier(X_train, y_train, w)
margins_sorted = np.sort(margins)

plt.figure(figsize=(10, 5))
plt.plot( margins_sorted,label="Отступ")
plt.axhline(0,color="red", linestyle="--", label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы объектов до обучения")
plt.legend()
plt.grid()
plt.show()

# Непрерывные аппроксимации пороговой функции потерь
# Квадратичная функция
def quadratic_loss(x_i, y_i, w):
    x_i = np.asarray(x_i, dtype=float)
    margin = margin_classifier(x_i, y_i, w)
    return (1 - margin) ** 2

# Реализовать вычисление градиента функции потерь
def gradient_loss(x_i, y_i, w):
  x_i = np.asarray(x_i, dtype=float)
  margin = margin_classifier(x_i, y_i, w)
  gradient = -2 * (1-margin) *y_i * x_i #это производная внешней функции
  return gradient

# Рекурентая оценка функционала качества
def update_quality(Q, loss, lambda_): #рекуррентное обновление оценки функционала качества
  return lambda_ * loss + (1 - lambda_) * Q

# loss — ошибка на текущем объекте
# Q — предыдущая оценка общей ошибки
# lambda_ — насколько сильно учитывать новую ошибку

# Реализовать L2 регуляризацию 
def l2_loss(loss, w, tau):
  return loss + (tau / 2) * np.sum(w ** 2)

# градиент
def l2_gradient(gradient, w, tau):  # добавляет L2-компонент к градиенту
  return gradient + tau * w

# Реализовать скорейший градиентный спуск
def steepest_gradient_step(x_i):
    x_i = np.asarray(x_i, dtype=float)
    x_2 = np.sum(x_i**2)
    if x_2 == 0:
          raise ValueError("Norm is zero")
    return 1 /(x_2)

# Реализовать предъявление объектов по модулю отступа (п.8):
# чем меньше |M_i|, тем больше вероятность выбрать объект
def choose_by_margin(X, y, w, rng, eps=1e-8):
    margins = np.abs(margin_classifier(X, y, w))
    probs = 1 / (margins + eps) #вес объекта при выборе
    probs = probs / probs.sum() #чем ближе объект к границе, тем чаще мы хотим его показывать модели
    return rng.choice(len(X), p=probs)


# Реализовать метод стохастического градиентного спуска с инерцией
def SGD(X, y, w, h, lambda_, gamma, q_sz, eps, max_iter=10000, by_margin = False, tau=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float).copy()
 
    rng = np.random.default_rng(42)
 
    # Инициализировать оценку функционала
    # как среднее L(w, x_i) по случайному подмножеству
    Q_ind = rng.choice(len(X), size=min(q_sz, len(X)), replace=False)
 
    losses = []
 
    for i in Q_ind:
        loss = l2_loss(quadratic_loss(X[i], y[i], w), w, tau)
        losses.append(loss)
 
    Q = np.mean(losses)
 
    # Начальное значение инерции
    v = np.zeros_like(w)
 
    for _ in range(max_iter):
 
        # Сохраняем предыдущие значения
        w_o = w.copy()
        Q_o = Q
 
        # Выбрать объект случайным образом
        if by_margin:
          i = choose_by_margin(X, y, w, rng)
        else:
            i = rng.integers(0, len(X))
 
        x_i = X[i]
        y_i = y[i]
 
        # ξ_i := L(w, x_i) (с L2-штрафом)
        e_i = l2_loss(quadratic_loss(x_i, y_i, w), w, tau)
 
        # delta L(w, x_i) (с L2-компонентой)
        gradient = l2_gradient(gradient_loss(x_i, y_i, w), w, tau)
 
        # Momentum:
        # v := γv + (1 - γ)∇L(w, x_i) # учитывает предыдущию градиенты
        v = gamma * v + (1 - gamma) * gradient
 
        # w := w - hv
        w = w - h * v
 
        if not np.all(np.isfinite(w)):
          print("weights diverged")
          break
 
        # Q := λξ_i + (1 - λ)Q
        Q = update_quality(Q, e_i, lambda_)
 
        # Проверка сходимости
        if (
            np.linalg.norm(w - w_o) < eps
            and abs(Q - Q_o) < eps
        ):
            break
 
    return w
 
# Скорейший градиентный спуск
def SGD_steepest(X, y, w, lambda_, q_sz, eps, max_iter, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float).copy()
 
    rng = np.random.default_rng(random_state)
 
    # Q
    Q_ind = rng.choice(len(X), size=min(q_sz, len(X)),replace=False)
 
    losses = []
 
    for i in Q_ind:
        losses.append(quadratic_loss( X[i], y[i], w))
 
    Q = np.mean(losses)
 
    for _ in range(max_iter):
 
        w_old = w.copy()
        Q_old = Q
 
        # случайный объект
        i = rng.integers(
            0,
            len(X)
        )
 
        x_i = X[i]
        y_i = y[i]
 
        e_i = quadratic_loss(x_i, y_i, w)
 
        gradient = gradient_loss(x_i, y_i,w)
 
        # h* = ||x_i||^-2
        h_i = steepest_gradient_step(x_i)
 
        w = w - h_i * gradient
 
        Q = update_quality(Q, e_i,lambda_)
 
        if (np.linalg.norm(w - w_old) < eps
            and abs(Q - Q_old) < eps):
            break
 
    return w, Q

norms = np.sum(np.asarray(X_train) ** 2, axis=1)

# print("Минимальная норма^2:", norms.min())
# print("Максимальная норма^2:", norms.max())
# print("Средняя норма^2:", norms.mean())
# print("Среднее h*:", np.mean(1 / norms))

# Обучение
#Обучить с инициализацией весов через корреляцию
def weights_correlation(X, y):
  numerator = X.T @ y
  denominator = np.sum(X ** 2, axis=0)
  w = np.zeros(X.shape[1])
  mask = denominator != 0
  w[mask] = numerator[mask] / denominator[mask]
  return w

w_corr = weights_correlation(X_train, y_train)

print("Минимальный вес:", np.min(w_corr))
print("Максимальный вес:", np.max(w_corr))
print("Норма вектора весов:", np.linalg.norm(w_corr))

margins = margin_classifier(X_train, y_train, w_corr)

print("Минимальный отступ:", np.min(margins))
print("Максимальный отступ:", np.max(margins))

w_corr = weights_correlation(X_train, y_train)

w_corr = SGD(X_train, y_train,w_corr,h,lambda_,gamma,q_sz,eps, tau=tau)

margins_corr_trained = np.sort(margin_classifier(X_train, y_train, w_corr))
 
plt.figure(figsize=(10, 5))
plt.plot(margins_corr_trained, label="Отступ")
plt.axhline(0, color="red", linestyle="--", label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы после обучения (инициализация через корреляцию)")
plt.legend()
plt.grid()
plt.show()

# Обучить со случайной инициализацией весов через мультистарт
n_starts = 5

best_w = None
best_loss = np.inf

for start in range(n_starts):

    # инициализация весов
    w = np.random.uniform(
        -1 / (2 * n_features),
        1 / (2 * n_features),
        size=n_features
    )

    # обучение
    w_trained = SGD(X_train, y_train, w, h, lambda_, gamma, q_sz, eps,tau=tau)

    # считаем среднюю квадратичную ошибку после обучения
    margins = margin_classifier( X_train, y_train, w_trained)

    current_loss = l2_loss(np.mean((1-margins)**2), w_trained, tau)

    if current_loss < best_loss:
        best_loss = current_loss
        best_w = w_trained.copy()

best_margins = margin_classifier(X_train,y_train,best_w)

sorted_best_margins = np.sort(best_margins)
plt.figure(figsize=(10, 5))
plt.plot(sorted_best_margins,label="Отступ")
plt.axhline(y=0, color = 'red',linestyle="--",label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы после обучения (мультистарт)")
plt.legend()
plt.grid()
plt.show()

# Обучить со случайным предъявлением и с п.8
w_start = np.random.uniform(
    -1 / (2 * n_features),
    1 / (2 * n_features),
    size=n_features
)

# без отступа
w_random = SGD(X_train,y_train,w_start.copy(),h,lambda_,gamma,q_sz,eps,by_margin=False)

#с отступом
w_margin = SGD(X_train,y_train,w_start.copy(),h,lambda_,gamma,q_sz,eps,by_margin=True)

margins_random = margin_classifier(X_train,y_train,w_random)

margins_margin = margin_classifier(X_train,y_train,w_margin)

loss_random = np.mean((1 - margins_random) ** 2)

loss_margin = np.mean((1 - margins_margin) ** 2)

# print("Ошибка при случайном предъявлении:", loss_random)
# print("Ошибка при предъявлении по модулю отступа:", loss_margin)

w_steepest, Q_steepest = SGD_steepest(
    X_train,
    y_train,
    w_start.copy(),
    lambda_,
    q_sz,
    eps,
    max_iter=10000
) 

margins_steepest = np.sort(margin_classifier(X_train, y_train, w_steepest))
loss_steepest = np.mean((1 - margins_steepest) ** 2)
# print("Ошибка при скорейшем градиентном спуске:", loss_steepest)

margins_random_sorted = np.sort(margins_random)
margins_margin_sorted = np.sort(margins_margin)

plt.figure(figsize=(10, 5))
plt.plot(margins_random_sorted,label="Рандомная выборка")
plt.plot(margins_margin_sorted,label="By |margin|")

# Граница ошибок:
# M < 0 — объект классифицирован неправильно
plt.axhline(0, linestyle="--")
plt.xlabel("Сортировка по отступу")
plt.ylabel("Margin")
plt.title("Рандомное предъявление vs предъявление by |margin|")
plt.legend()
plt.grid()
plt.show()

# Оценим качество классификации
def predict(X, w):
    scores = X @ w
    return np.where(scores >= 0, 1, -1)

def evaluate(X, y, w):
    y_pred = predict(X, w)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label=1)
    recall = recall_score(y, y_pred, pos_label=1)
    f1 = f1_score(y, y_pred, pos_label=1)

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    return accuracy, precision, recall, f1

print("Инициализация через корреляцию:")
evaluate(X_test, y_test, w_corr)

print("\VМультистарт:")
evaluate(X_test, y_test, best_w)

print("\nСлучайная инициализация:")
evaluate(X_test, y_test, w_random)

print("\nС отступом:")
evaluate(X_test, y_test, w_margin)

#Сравним с квадратичной функцией потерь
def mean_loss(X, y, w):
    margins = margin_classifier(X, y, w)
    return np.mean((1 - margins) ** 2)

models = {
    "Correlation": w_corr,
    "Multistart": best_w,
    "Random presentation": w_random,
    "Margin presentation": w_margin,
    "Скорейший спуск": w_steepest
}

train_losses = {}

for name, weights in models.items():
    loss = mean_loss(X_train, y_train, weights)
    train_losses[name] = loss
    print(name, ":", loss)

best_model_name = min(
    train_losses,
    key=train_losses.get
)

best_own_w = models[best_model_name]

print("\nЛучшая собственная модель:", best_model_name)
print("Ошибка на обучении:", train_losses[best_model_name])
 
print("\nКачество лучшей собственной модели на тесте:")
own_metrics = evaluate(X_test,y_test,best_own_w)
print(own_metrics)

# Сравнить лучшую реализацию с эталонной
from sklearn.linear_model import RidgeClassifier
 
reference_model = RidgeClassifier(alpha=0.0,fit_intercept=False)
 
reference_model.fit(X_train,y_train)
 
y_pred_reference = reference_model.predict(X_test)
 
reference_accuracy = accuracy_score( y_test, y_pred_reference)
 
reference_precision = precision_score( y_test, y_pred_reference, pos_label=1)
 
reference_recall = recall_score( y_test, y_pred_reference, pos_label=1)
 
reference_f1 = f1_score( y_test, y_pred_reference,pos_label=1)


# Метрики качества модели
y_pred_own = predict(X_test, best_own_w)
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
metrics_values = own_metrics
plt.figure(figsize=(7, 4))
plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1)
plt.title("Качество модели")
plt.ylabel("Значение")
plt.show()

# мтрица ошибок 
cm = confusion_matrix(y_test, y_pred_own)
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.xticks([0, 1], ["-1", "1"])
plt.yticks([0, 1], ["-1", "1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()

print("Эталонная модель:")
print("Accuracy (доля верных ответов):", reference_accuracy)
print("Precision (точность):", reference_precision)
print("Recall (полнота):", reference_recall)
print("F1-мера:", reference_f1)
 
print("\nЛучшая собственная модель:", best_model_name)
 
print("\nСобственная реализация:")
print("Accuracy (доля верных ответов):", own_metrics[0])
print("Precision (точность):", own_metrics[1])
print("Recall (полнота):", own_metrics[2])
print("F1-мера:", own_metrics[3])
 
print("\nЭталонная реализация:")
print("Accuracy (доля верных ответов):", reference_accuracy)
print("Precision (точность):", reference_precision)
print("Recall (полнота):", reference_recall)
print("F1-мера:", reference_f1)
 
