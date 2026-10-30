import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from pathlib import Path
from sklearn.linear_model import RidgeClassifier


IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

url = (
    "https://raw.githubusercontent.com/"
    "mohitbatheja/diabetes-risk-prediction-multinomial-logistic-regression/"
    "main/diabetes_risk.csv")

data = pd.read_csv(url)

h = 1e-3 # шаг
lambda_ = 0.001 # какая доля новой ошибки учитывается в новом Q
gamma = 0.9 # насколько сильно учитывается предыдущая инерция
q_sz = 100 # выбираются примерно 100 случайных объектов 
eps = 1e-6 #порог чувствительности
tau = 1e-3 #l2 регуляризация
check_every = 200 # проверяем сходимость

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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42,stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

mode_columns = ["alcohol_consumption", "smoking_status", "income_bracket"]
for column in mode_columns:
    train_mode = X_train[column].mode(dropna=True)
    fill_value = train_mode.iloc[0]
    X_train[column] = X_train[column].fillna(fill_value)
    X_val[column] = X_val[column].fillna(fill_value)  
    X_test[column] = X_test[column].fillna(fill_value)
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
# print(categorical_columns)

num_columns = X.select_dtypes(include="number").columns.tolist()
mu = X_train[num_columns].mean()
sigma = X_train[num_columns].std().replace(0, 1)
X_train[num_columns] = (X_train[num_columns] - mu) / sigma
X_val[num_columns] = (X_val[num_columns] - mu) / sigma  
X_test[num_columns] = (X_test[num_columns] - mu) / sigma

X_train = pd.get_dummies(X_train, columns=categorical_columns,drop_first=True)
X_val = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)  
X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0) 
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.astype(float)
X_val = X_val.astype(float)  
X_test = X_test.astype(float)

X_train["bias"] = -1.0
X_val["bias"] = -1.0  
X_test["bias"] = -1.0

n_features = X_train.shape[1]

rng = np.random.default_rng(42)

w = rng.uniform(
    -1 / (2 * n_features),
    1 / (2 * n_features),
    size=n_features
)

# Реализовать вычисление отступа объекта (визуализировать, проанализировать)
def margin_classifier(X, y, w):
  X = np.asarray(X)
  y = np.asarray(y)
  return y * (X @ w) 

def margin_stats(name, X, y, w):
    m = margin_classifier(X, y, w)
    print(f"{name}: доля M<0 = {np.mean(m < 0):.4f}, доля |M|<0.1 = {np.mean(np.abs(m) < 0.1):.4f}, "
          f"mean = {m.mean():.4f}, min = {m.min():.4f}, max = {m.max():.4f}")

margins = margin_classifier(X_train, y_train, w)
margin_stats("До обучения", X_train, y_train, w) #
margins_sorted = np.sort(margins)
 
plt.figure(figsize=(10, 5))
plt.plot( margins_sorted,label="Отступ")
plt.axhline(0,color="red", linestyle="--", label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы объектов до обучения")
plt.legend()
plt.grid()
plt.savefig(IMAGES_DIR / "objects_before_learning.png", dpi=150, bbox_inches="tight")
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

# Рекуррентная оценка функционала качества
def update_quality(Q, loss, lambda_): #рекуррентное обновление оценки функционала качества
  return lambda_ * loss + (1 - lambda_) * Q

# loss — ошибка на текущем объекте
# Q — предыдущая оценка общей ошибки
# lambda_ — насколько сильно учитывать новую ошибку

# Реализовать L2 регуляризацию 
def l2_loss(loss, w, tau):
    return loss + (tau / 2) * np.sum(w[:-1] ** 2)

# градиент
def l2_gradient(gradient, w, tau):
    regularization = tau * w.copy()
    regularization[-1] = 0.0
    return gradient + regularization

# Реализовать скорейший градиентный спуск
def steepest_gradient_step(x_i):
    x_i = np.asarray(x_i, dtype=float)
    x_2 = np.sum(x_i ** 2)
    if x_2 == 0:
        raise ValueError("Norm is zero")
    return 1 / (2 * x_2) # исправила формулу

# Реализовать предъявление объектов по модулю отступа (п.8):
# чем меньше |M_i|, тем больше вероятность выбрать объект
def choose_by_margin(X, y, w, rng, eps=1e-8):
    margins = np.abs(margin_classifier(X, y, w))
    probs = 1 / (margins + eps) #вес объекта при выборе
    probs = probs / probs.sum() #чем ближе объект к границе, тем чаще мы хотим его показывать модели
    return rng.choice(len(X), p=probs)


# Реализовать метод стохастического градиентного спуска с инерцией
# в него записывается значение Q на каждой итерации (нужно для графика Q)
def SGD(X, y, w, h, lambda_, gamma, q_sz, eps, max_iter=40000, by_margin = False, tau=0, seed=42, Q_history=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float).copy()
 
    rng = np.random.default_rng(seed)
 
    # Инициализировать оценку функционала
    # как среднее L(w, x_i) по случайному подмножеству
    Q_ind = rng.choice(len(X), size=min(q_sz, len(X)), replace=False)
 
    losses = []
 
    for i in Q_ind:
        loss = l2_loss(quadratic_loss(X[i], y[i], w), w, tau)
        losses.append(loss)

    Q = np.mean(losses)
    v = np.zeros_like(w)

    w_o = w.copy()
    Q_o = Q

    for n_iter in range(1, max_iter + 1): 
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
        # v := γv + (1 - γ)∇L(w, x_i) # учитывает предыдущие градиенты  
        v = gamma * v + (1 - gamma) * gradient
 
        # w := w - hv
        w = w - h * v
 
        if not np.all(np.isfinite(w)):
          print("weights diverged")
          break
 
        # Q := λξ_i + (1 - λ)Q
        Q = update_quality(Q, e_i, lambda_)
        if Q_history is not None:  
            Q_history.append(Q)

        if n_iter % check_every == 0:
            if (
                np.linalg.norm(w - w_o) < eps
                and abs(Q - Q_o) < eps
            ):
                break
            w_o = w.copy()
            Q_o = Q
 
    print("SGD: итераций =", n_iter) 
    return w
 
# Скорейший градиентный спуск
def SGD_steepest(X, y, w, lambda_, q_sz, eps, max_iter, tau=0, random_state=42, Q_history=None): 
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float).copy()
 
    rng = np.random.default_rng(random_state)
 
    # Q
    Q_ind = rng.choice(len(X), size=min(q_sz, len(X)),replace=False)
 
    losses = []
 
    for i in Q_ind:
        losses.append(l2_loss(quadratic_loss( X[i], y[i], w), w, tau))
 
    Q = np.mean(losses)
 
    w_old = w.copy() 
    Q_old = Q
 
    for n_iter in range(1, max_iter + 1): 
 
        # случайный объект
        i = rng.integers(
            0,
            len(X)
        )
 
        x_i = X[i]
        y_i = y[i]
 
        e_i = l2_loss(quadratic_loss(x_i, y_i, w), w, tau)
 
        gradient = l2_gradient(gradient_loss(x_i, y_i,w), w, tau) 
 
        # h* = 1 / (2 ||x_i||^2) 
        h_i = steepest_gradient_step(x_i)
 
        w = w - h_i * gradient
 
        Q = update_quality(Q, e_i,lambda_)
        if Q_history is not None:  
            Q_history.append(Q)

        if n_iter % check_every == 0:
            if (np.linalg.norm(w - w_old) < eps
                and abs(Q - Q_old) < eps):
                break
            w_old = w.copy()
            Q_old = Q
        
 
    print("Скорейший спуск: итераций =", n_iter) 
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

w_corr = SGD(X_train, y_train,w_corr,h,lambda_,gamma,q_sz,eps, tau=tau)

margins_corr_trained = np.sort(margin_classifier(X_train, y_train, w_corr))
margin_stats("Корреляция (после SGD)", X_train, y_train, w_corr) 
 
plt.figure(figsize=(10, 5))
plt.plot(margins_corr_trained, label="Отступ")
plt.axhline(0, color="red", linestyle="--", label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы после обучения (инициализация через корреляцию)")
plt.legend()
plt.grid()
plt.savefig(IMAGES_DIR / "initialization_with_correlation.png", dpi=150, bbox_inches="tight") 
plt.show()

# Обучить со случайной инициализацией весов через мультистарт
n_starts = 5

best_w = None
best_loss = np.inf

for start in range(n_starts):

    # инициализация весов
    w = np.random.default_rng(100 + start).uniform(
        -1 / (2 * n_features),
        1 / (2 * n_features),
        size=n_features
    )

    # обучение
    w_trained = SGD(X_train, y_train, w, h, lambda_, gamma, q_sz, eps,tau=tau, seed=start)

    # считаем среднюю квадратичную ошибку после обучения
    margins = margin_classifier( X_train, y_train, w_trained)

    current_loss = l2_loss(np.mean((1-margins)**2), w_trained, tau)

    if current_loss < best_loss:
        best_loss = current_loss
        best_w = w_trained.copy()

best_margins = margin_classifier(X_train,y_train,best_w)
margin_stats("Мультистарт (лучший)", X_train, y_train, best_w)

sorted_best_margins = np.sort(best_margins)
plt.figure(figsize=(10, 5))
plt.plot(sorted_best_margins,label="Отступ")
plt.axhline(y=0, color = 'red',linestyle="--",label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы после обучения (мультистарт)")
plt.legend()
plt.grid()
plt.savefig(IMAGES_DIR / "multistart.png", dpi=150, bbox_inches="tight")  
plt.show()

# Обучить со случайным предъявлением и с п.8
w_start = np.random.default_rng(7).uniform(
    -1 / (2 * n_features),
    1 / (2 * n_features),
    size=n_features
)

Q_hist_random, Q_hist_margin, Q_hist_steepest = [], [], []

# без отступа
w_random = SGD(X_train,y_train,w_start.copy(),h,lambda_,gamma,q_sz,eps,by_margin=False,tau=tau, Q_history=Q_hist_random)

#с отступом
w_margin = SGD(X_train,y_train,w_start.copy(),h,lambda_,gamma,q_sz,eps,by_margin=True,tau=tau, Q_history=Q_hist_margin)

abs_m = np.abs(margin_classifier(X_train, y_train, w_margin))
p_choice = 1 / (abs_m + 1e-8)
p_choice = np.sort(p_choice / p_choice.sum())
print(f"Предъявление по |M|: макс. вероятность выбора одного объекта = {p_choice[-1]:.4f}, "
      f"сумма вероятностей 10 самых частых объектов = {p_choice[-10:].sum():.4f}")

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
    max_iter=40000,
    tau=tau,
    Q_history=Q_hist_steepest  
)

margins_steepest = np.sort(margin_classifier(X_train, y_train, w_steepest))
loss_steepest = np.mean((1 - margins_steepest) ** 2)
# print("Ошибка при скорейшем градиентном спуске:", loss_steepest)

margins_random_sorted = np.sort(margins_random)
margins_margin_sorted = np.sort(margins_margin)

plt.figure(figsize=(10, 5))
plt.plot(margins_random_sorted,label="Случайное предъявление")
plt.plot(margins_margin_sorted,label="Предъявление по модулю отступа")

# Граница ошибок:
# M < 0 — объект классифицирован неправильно
plt.axhline(0, color="red", linestyle="--", label="M = 0")
plt.xlabel("Объекты, отсортированные по отступу")
plt.ylabel("Отступ")
plt.title("Отступы после обучения: случайное предъявление и предъявление по |M|")
plt.legend()
plt.grid()
plt.savefig(IMAGES_DIR / "margin.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(Q_hist_random, label="SGD с инерцией, случайное предъявление")
plt.plot(Q_hist_margin, label="SGD с инерцией, предъявление по |M|")
plt.plot(Q_hist_steepest, label="Скорейший спуск")
plt.xlabel("Итерация")
plt.ylabel("Q (рекуррентная оценка)")
plt.title("Рекуррентная оценка функционала качества Q")
plt.legend()
plt.grid()
plt.savefig(IMAGES_DIR / "quality_functional.png", dpi=150, bbox_inches="tight")
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

print("\nМультистарт:")
evaluate(X_test, y_test, best_w)

print("\nСлучайная инициализация:")
evaluate(X_test, y_test, w_random)

print("\nС отступом:")
evaluate(X_test, y_test, w_margin)

print("\nСкорейший спуск:")
evaluate(X_test, y_test, w_steepest)

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

print("\nОбученные пороги активации:")

for name, weights in models.items():
    print(f"{name}: w0 = {weights[-1]:.6f}")

train_losses = {}

for name, weights in models.items():
    loss = mean_loss(X_train, y_train, weights)
    train_losses[name] = loss
    print(name, ":", loss)

val_losses = {name: mean_loss(X_val, y_val, weights) for name, weights in models.items()}
 
print("\nОшибка на валидации:")
for name, loss in val_losses.items():
    print(name, ":", loss)
 
best_model_name = min(val_losses, key=val_losses.get)
best_own_w = models[best_model_name]

print("\nЛучшая собственная модель:", best_model_name)
print("Ошибка на обучении:", train_losses[best_model_name])

weights_table = pd.Series(best_own_w, index=X_train.columns).drop("bias")
print("\nНаибольшие по модулю веса лучшей модели:")
print(weights_table.reindex(weights_table.abs().sort_values(ascending=False).index).head(8).round(3))
city_w = weights_table[[c for c in weights_table.index if c.startswith("city_")]]
print(f"Веса city_*: от {city_w.min():.3f} до {city_w.max():.3f}")
 
print("\nКачество лучшей собственной модели на тесте:")
own_metrics = evaluate(X_test,y_test,best_own_w)
print(own_metrics)

# Сравнить лучшую реализацию с эталонной
 
reference_model = RidgeClassifier(alpha=0.0,fit_intercept=False)
 
reference_model.fit(X_train,y_train)
 
y_pred_reference = reference_model.predict(X_test)
 
reference_accuracy = accuracy_score( y_test, y_pred_reference)
 
reference_precision = precision_score( y_test, y_pred_reference, pos_label=1)
 
reference_recall = recall_score( y_test, y_pred_reference, pos_label=1)
 
reference_f1 = f1_score( y_test, y_pred_reference,pos_label=1)

# RidgeClassifier минимизирует СУММУ квадратов + alpha*||w||^2,
alpha_equiv = tau * len(X_train) / 2
reference_reg = RidgeClassifier(alpha=alpha_equiv, fit_intercept=False).fit(X_train, y_train)
y_pred_reference_reg = reference_reg.predict(X_test)
print(f"\nЭталон с alpha = tau*l/2 = {alpha_equiv:.2f}: "
      f"Accuracy = {accuracy_score(y_test, y_pred_reference_reg):.4f}, "
      f"F1 = {f1_score(y_test, y_pred_reference_reg, pos_label=1):.4f}")


# Метрики качества модели
y_pred_own = predict(X_test, best_own_w)
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
reference_metrics = (reference_accuracy, reference_precision, reference_recall, reference_f1)
x_pos = np.arange(len(metrics_names))
bar_w = 0.38
plt.figure(figsize=(8, 4.5))
bars_own = plt.bar(x_pos - bar_w / 2, own_metrics, bar_w, label=f"Собственная модель ({best_model_name})")
bars_ref = plt.bar(x_pos + bar_w / 2, reference_metrics, bar_w, label="Эталон (RidgeClassifier)")
for bars in (bars_own, bars_ref):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                 ha="center", va="bottom", fontsize=9)
plt.xticks(x_pos, metrics_names)
plt.ylim(0, 1.05)
plt.title("Качество на тестовой выборке: собственная модель и эталон")
plt.ylabel("Значение метрики")
plt.legend(loc="lower right")
plt.savefig(IMAGES_DIR / "quality_model.png", dpi=150, bbox_inches="tight")
plt.show()

# матрица ошибок
cm = confusion_matrix(y_test, y_pred_own)
plt.figure(figsize=(5.5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Число объектов")
plt.title(f"Матрица ошибок ({best_model_name}, тест)")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.xticks([0, 1], ["-1", "1"])
plt.yticks([0, 1], ["-1", "1"])
for i in range(2):
    for j in range(2):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center", color=text_color)
plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
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
 
