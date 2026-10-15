import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # эталонная реализация
from sklearn.decomposition import PCA


# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
data = pd.read_csv('/Users/arina/Desktop/university/fall-2026/students/astapenkova-av/lab2/source/heart.csv', delimiter=',')
# размер датасета
# print(data.shape)

#print(data.info)

# описание данных 
#print(data.describe())

#print(data.duplicated().sum()) # ноль дублей

# категориальные данные
print(data.columns.tolist())

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

# 2. реализовать алгоритм KNN с методом окна Парзена переменной ширины:
def minkowski_distance(X, x, p=2): 
    diff = np.abs(X - x)
    if p == np.inf:
        return np.max(diff, axis=1)
    return np.sum(diff ** p, axis=1) ** (1.0 / p)

# 1. в качестве ядра можно использовать гауссово ядро;
def gausen_kernel(r): #ширина окна влияет на точность аппроксимации
    return np.exp(-2.0 * r **2) 

def parzen_window(X_train, y_train, x, k, p=2, kernel=gausen_kernel):
    distance = minkowski_distance(X_train, x, p=p)
    order = np.argsort(distance)
    sorted_dists = distance[order]
    sorted_labels = y_train[order]
    h = sorted_dists[k]
    if h == 0:
        h = 1e-12
    weights = kernel(sorted_dists / h)
    classes = np.unique(y_train)
    scores = {}
    for c in classes:
        scores[c] = np.sum(weights[sorted_labels == c])
    return max(scores, key=scores.get)

def parzen_predict(X_train, y_train, X_test, k, p=2, kernel=gausen_kernel):

    predictions = []
    for x in X_test:

        prediction = parzen_window(X_train, y_train, x, k, p=p, kernel=kernel)
        predictions.append(prediction)

    return np.array(predictions)

def loo(X, y, k, p=2, kernel=gausen_kernel):
    errors = 0
    n = len(X)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_loo = X[mask]
        y_loo = y[mask]
        prediction = parzen_window(X_loo, y_loo, X[i], k, p=p, kernel=kernel)

        if prediction != y[i]:
            errors += 1
    return errors / n

def best_k(X, y, k_v, p=2, kernel=gausen_kernel):
    loo_errors=[]
    for k in k_v:
        error = loo(X, y, k, p=p, kernel=kernel)
        loo_errors.append(error)

    loo_errors = np.array(loo_errors)
    best_index = np.argmin(loo_errors)
    optimal = k_v[best_index]
    return optimal, loo_errors

# для визуализации 
def parzen_score_diff_batch(X_ref, y_ref, Q, k, p=2, kernel=gausen_kernel, chunk=500):
    out = []
    for start in range(0, len(Q), chunk):
        D = pairwise_dist(Q[start:start + chunk], X_ref, p)
        order = np.argsort(D, axis=1)
        sd = np.take_along_axis(D, order, axis=1)
        sl = y_ref[order]
        h = sd[:, k]
        h = np.where(h == 0, 1e-12, h)
        W = kernel(sd / h[:, None])
        out.append((W * (sl == 1)).sum(axis=1) - (W * (sl == 0)).sum(axis=1))
    return np.concatenate(out)
 
def draw_decision_map(ax, X_ref, y_ref, pca, lims, k, X_pts, y_pts, title,
                      p=2, kernel=gausen_kernel, grid_size=100, pt_size=20):
    xs = np.linspace(lims[0], lims[1], grid_size)
    ys = np.linspace(lims[2], lims[3], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_full = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = parzen_score_diff_batch(X_ref, y_ref, grid_full, k, p=p, kernel=kernel).reshape(xx.shape)
    lim = np.abs(Z).max()
    ax.imshow(Z, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
              cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.scatter(X_pts[:, 0], X_pts[:, 1], c=np.where(y_pts == 1, "lime", "red"),
               edgecolors="black", s=pt_size)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)


# 3. подобрать параметр k методом скользящего контроля (LOO)
# 4. обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;

# 5. сравнить с [эталонной](https://scikit-learn.org/stable/) реализацией KNN;
#    1. сравнить качество работы алгоритмов;
# 6. реализовать алгоритм отбора эталонов;
# Алгоритм STOLP для отбора эталонных объектов
def pairwise_dist(A, B, p=2):
    diff = np.abs(A[:, None, :] - B[None, :, :])
    if p == np.inf:
        return np.max(diff, axis=2)
    return np.sum(diff ** p, axis=2) ** (1.0 / p)

def parzen_margins(D, y_ref, y_query, k, n_valid, kernel=gausen_kernel):
    kk = min(k, n_valid - 1)
    order = np.argsort(D, axis=1)
    sd = np.take_along_axis(D, order, axis=1)
    sl = y_ref[order]
    h = sd[:, kk]
    h = np.where(h == 0, 1e-12, h)
    W = kernel(sd / h[:, None])
    classes = np.unique(y_ref)
    G = np.stack([(W * (sl == c)).sum(axis=1) for c in classes], axis=1)
    own = classes[None, :] == y_query[:, None]
    g_own = (G * own).sum(axis=1)
    g_other = np.where(own, -np.inf, G).max(axis=1)
    total = np.where(G.sum(axis=1) == 0, 1e-12, G.sum(axis=1))
    return (g_own - g_other) / total

def stolp(X, y, k, delta=0.5, l0=10, p=2, kernel=gausen_kernel):
    n = len(X)
    classes = np.unique(y)
    D = pairwise_dist(X, X, p)
 
    D_loo = D.copy()
    np.fill_diagonal(D_loo, np.inf)
    margins = parzen_margins(D_loo, y, y, k, n_valid=n - 1, kernel=kernel)
    noise = np.where(margins < -delta)[0]
    keep = np.where(margins >= -delta)[0]
 
    omega = []
    for c in classes:
        idx = keep[y[keep] == c]
        omega.append(idx[np.argmax(margins[idx])])
 
    history = []
    while True:
        rest = np.setdiff1d(keep, omega)
        if len(rest) == 0:
            break
        m = parzen_margins(D[np.ix_(rest, omega)], y[omega], y[rest], k,
                           n_valid=len(omega), kernel=kernel)
        n_err = int(np.sum(m < 0))
        history.append(n_err)
        if n_err <= l0:
            break
        omega.append(rest[np.argmin(m)])
 
    return np.array(omega), noise, history
# 7. подготовить визуализацию результатов работы алгоритма отбора эталонов;
# 8. сравнить качество работы KNN с и без отбора эталонов; 
# 7. визуализация отбора эталонов

if __name__ == "__main__":
    k_values = list(range(1, 50))
    optimal_k, losses = best_k(X_train, y_train, k_values, p=2, kernel=gausen_kernel)
    print(f"Лучшее k по LOO: {optimal_k}, LOO-ошибка: {losses[optimal_k - 1]:.4f}")
 
    # эмпирический риск на обучающей выборке для каждого k
    train_errors = [
        np.mean(parzen_predict(X_train, y_train, X_train, k) != y_train)
        for k in k_values
    ]
 
    # график 2: частота ошибок от числа соседей
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, losses, color="red", label="LOO (скользящий контроль)")
    plt.plot(k_values, train_errors, color="blue", label="эмпирический риск (обучение)")
    plt.axvline(optimal_k, color="gray", linestyle="--", label=f"оптимальное k = {optimal_k}")
    plt.xticks(k_values, fontsize=8)      
    plt.xlabel("число соседей k")
    plt.ylabel("частота ошибок")
    plt.title("Зависимость ошибки от числа соседей")
    plt.legend()
    plt.grid(True)
    plt.savefig("loo_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # PCA для визуализации
    pca = PCA(n_components=2).fit(X_train)
    X2 = pca.transform(X_train)
    pad = 0.5
    lims = (X2[:, 0].min() - pad, X2[:, 0].max() + pad, X2[:, 1].min() - pad, X2[:, 1].max() + pad)
 
    # качество собственной реализации
    y_pred_custom = parzen_predict(X_train, y_train, X_test, optimal_k, p=2, kernel=gausen_kernel)
    print(f"Точность Парзена (k={optimal_k}): {np.mean(y_pred_custom == y_test):.4f}")
 
    # эталонная реализация sklearn
    ref_model = KNeighborsClassifier(n_neighbors=optimal_k)
    ref_model.fit(X_train, y_train)
    print(f"Точность sklearn KNN (k={optimal_k}): {ref_model.score(X_test, y_test):.4f}")
 
 
    omega, noise, history = stolp(X_train, y_train, optimal_k, delta=0.5, l0=10)
    X_proto, y_proto = X_train[omega], y_train[omega]
    print(f"\nSTOLP: эталонов {len(omega)} из {len(X_train)}, выбросов {len(noise)}")
 
    k_proto_values = list(range(1, min(50, len(omega) - 1)))
    k_proto, _ = best_k(X_proto, y_proto, k_proto_values, p=2, kernel=gausen_kernel)
 
    y_pred_proto = parzen_predict(X_proto, y_proto, X_test, k_proto, p=2, kernel=gausen_kernel)
    acc_parzen_proto = np.mean(y_pred_proto == y_test)
 
    ref_proto = KNeighborsClassifier(n_neighbors=min(k_proto, len(omega)))
    ref_proto.fit(X_proto, y_proto)
    acc_ref_proto = ref_proto.score(X_test, y_test)
 
    print(f"{'':28}{'без отбора':>12}{'с отбором':>12}")
    print(f"{'Парзен (свой)':28}{np.mean(y_pred_custom == y_test):12.4f}{acc_parzen_proto:12.4f}")
    print(f"{'sklearn KNN':28}{ref_model.score(X_test, y_test):12.4f}{acc_ref_proto:12.4f}")
    print(f"{'k':28}{optimal_k:12d}{k_proto:12d}")
    print(f"{'размер обучающей выборки':28}{len(X_train):12d}{len(omega):12d}")
 
    # график 7: результат отбора эталонов (до / отобранные / после)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    draw_decision_map(axes[0], X_train, y_train, pca, lims, optimal_k, X2, y_train,
                      f"Без отбора: {len(X_train)} объектов, k = {optimal_k}", pt_size=14)
 
    is_proto = np.zeros(len(y_train), dtype=bool)
    is_proto[omega] = True
    is_noise = np.zeros(len(y_train), dtype=bool)
    is_noise[noise] = True
    rest = ~is_proto & ~is_noise
    ax = axes[1]
    ax.scatter(X2[rest, 0], X2[rest, 1], c=np.where(y_train[rest] == 1, "lime", "red"),
               s=14, alpha=0.35, label="остальные")
    ax.scatter(X2[is_noise, 0], X2[is_noise, 1], marker="x", c="black", s=40,
               label=f"выбросы ({is_noise.sum()})")
    ax.scatter(X2[is_proto, 0], X2[is_proto, 1], c=np.where(y_train[is_proto] == 1, "lime", "red"),
               edgecolors="black", linewidths=1.2, s=70, label=f"эталоны ({is_proto.sum()})")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Результат отбора эталонов (STOLP)")
    ax.legend()
 
    draw_decision_map(axes[2], X_proto, y_proto, pca, lims, k_proto, X2[omega], y_proto,
                      f"С отбором: {len(omega)} эталонов, k = {k_proto}", pt_size=40)
    fig.savefig("stolp_result.png", dpi=150, bbox_inches="tight")
    plt.show()
 
