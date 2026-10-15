import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # эталонная реализация


# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
data = pd.read_csv('/Users/arlinrus/Desktop/fall-2026/students/astapenkova-av/lab2/source/heart.csv', delimiter=',')
# размер датасета
# print(data.shape)

#print(data.info)

# описание данных 
#print(data.describe())

#print(data.duplicated().sum()) # ноль дублей

# категориальные данные
#print(data.columns.tolist())

# пропущенные значения
#print(data.dtypes)

# разделить на вход и выход
target = 'HeartDisease'
X = data.drop(columns=[target])
categorial = X.select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(X, columns=categorial, drop_first=True, dtype=int)
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit только на train!
X_test = scaler.transform(X_test)   

# Training data shape: (734, 16)
# Testing data shape: (184, 16)
# print("\nТренировочная ввыборка:", X_train.shape)
# print("Тестовая выборка:", X_test.shape)

# 2. реализовать алгоритм KNN с методом окна Парзена переменной ширины;
def euklidian_distance(x,xi):
    return np.sqrt(np.sum((x-xi))**2)

# 1. в качестве ядра можно использовать гауссово ядро;
def gausen_kernel(r): #ширина окна влияет на точность аппроксимации
# не убывает при |r| -> inf к нулю, положительно и невозрастает на [0, 1]
    return np.exp(-2.0 * r **2)

def minkowski_distance(X, x, p=2): # при p=2 это евклидова метрика
    diff = np.abs(X - x) # разнится по модулю
    if p == np.inf:
        return np.max(diff, axis=1)
    pow = diff ** p
    sum = np.sum(pow, axis=1)
    return sum ** (1.0/p)


def parzen_one(X_train, y_train, x, k, classes, p=2, kernel=gausen_kernel): #классификация одной точки
    distance = minkowski_distance(X_train, x, p=p)
 
    order = np.argsort(distance)          # индексы объектов по возрастанию расстояния
    sorted_dists = distance[order]
    sorted_labels = y_train[order]
 
    h = sorted_dists[k]                # h(x) = rho(x, x^(k+1))
    if h == 0:
        h = 1e-9                       # защита от деления на 0 (совпадающие точки)
 
    weights = kernel(sorted_dists / h)  # w_i(x) = K( rho(x, x_i) / h(x) )
 
    # взвешенное голосование по классам: a(x) = argmax_y sum_i [y_i = y] * w_i(x)
    scores = {c: 0.0 for c in classes}
    for w, label in zip(weights, sorted_labels):
        scores[label] += w
 
    best_class, best_score = None, -np.inf
    for c, s in scores.items():
        if s > best_score:
            best_score, best_class = s, c
 
    return best_class

def parzen_predict(X_train, y_train, X_test, k, classes, p=2, kernel=gausen_kernel):  # ИСПРАВЛЕНО: kernel=gausen_kernel вместо kernel=kernel (было NameError)
    return np.array([
        parzen_one(X_train, y_train, x, k, classes, p=p, kernel=kernel)  # ИСПРАВЛЕНО: вызываем parzen_one (классификация одной точки), а не саму себя
        for x in X_test
    ])

# 3. подобрать параметр k методом скользящего контроля (LOO);
def loo_error(X, y, k, classes, p=2, kernel=gausen_kernel):
    """
    Ошибка LOO (Leave-One-Out) для заданного k — доля неверно
    классифицированных объектов при поочерёдном исключении каждого
    объекта из обучающей выборки.
 
        LOO(k) = (1/l) * sum_i [ a(x_i; X \\ {x_i}) != y_i ]
    """
    n = len(X)
    errors = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = parzen_one(X[mask], y[mask], X[i], k, classes, p=p, kernel=kernel)  # ИСПРАВЛЕНО: parzen_one, т.к. X[i] — одна точка, а не пакет
        if pred != y[i]:
            errors += 1
    return errors / n


def select_best_k(X, y, k_values, classes, p=2, kernel=gausen_kernel):
    losses = []
    for k in k_values:
        loss = loo_error(X, y, k, classes, p=p, kernel=kernel)
        losses.append(loss)
 
    losses = np.array(losses)
    best_idx = int(np.argmin(losses))
    return k_values[best_idx], losses

# 4. обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;

# 5. сравнить с [эталонной](https://scikit-learn.org/stable/) реализацией KNN;
#    1. сравнить качество работы алгоритмов;
# 6. реализовать алгоритм отбора эталонов;
# 7. подготовить визуализацию результатов работы алгоритма отбора эталонов;
# 8. сравнить качество работы KNN с и без отбора эталонов; 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
 
    classes = np.unique(y_train)
 
    # LOO лучше считать не на всей обучающей выборке, а на подвыборке,
    # т.к. это O(n^2) — иначе на больших датасетах будет очень медленно
    k_values = list(range(1, 50))
    best_k, losses = select_best_k(X_train, y_train, k_values, classes)
    print(f"Лучшее k по LOO: {best_k}, LOO-ошибка: {losses[best_k - 1]:.4f}")
 
    # график зависимости LOO-ошибки от k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, losses, marker="o", label="LOO-ошибка")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Оптимальное k = {best_k}")
    plt.xlabel("k (число соседей)")
    plt.ylabel("LOO-ошибка")
    plt.title("Зависимость LOO-ошибки от параметра k (heart.csv)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loo_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    # оценка качества собственной реализации на тестовой выборке
    y_pred_custom = parzen_predict(X_train, y_train, X_test, best_k, classes)  # ИСПРАВЛЕНО: parzen_predict вместо несуществующей parzen_predict_batch
    accuracy_custom = np.mean(y_pred_custom == y_test)
    print(f"Точность собственной реализации (Парзен, k={best_k}): {accuracy_custom:.4f}")
 
    # сравнение с эталонной реализацией sklearn
    ref_model = KNeighborsClassifier(n_neighbors=best_k)
    ref_model.fit(X_train, y_train)
    y_pred_ref = ref_model.predict(X_test)
    accuracy_ref = np.mean(y_pred_ref == y_test)
    print(f"Точность эталонной реализации (sklearn KNN, k={best_k}): {accuracy_ref:.4f}")