import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # эталонная реализация
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib.patches import Patch

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "benbobyabraham/heart_failure_prediction_dataset_kaggle/"
    "main/heart.csv"
)
data = pd.read_csv(DATA_URL)

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
categorial = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X = pd.get_dummies(X, columns=categorial, drop_first=True, dtype=int)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

X_train = X_train.to_numpy() 
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)   

# Training data shape: (734, 15)
# Testing data shape: (184, 15)
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
    weights[k:] = 0.0
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
        W[:, k:] = 0.0  
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
    im = ax.imshow(Z, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
                   cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for c, color, name in ((1, "lime", "класс 1 (болезнь)"), (0, "red", "класс 0 (нет болезни)")):
        mask = y_pts == c
        ax.scatter(X_pts[mask, 0], X_pts[mask, 1], c=color, edgecolors="black", s=pt_size, label=name)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Γ₁ − Γ₀\n(> 0: предсказан класс 1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

def pairwise_dist(A, B, p=2):
    diff = np.abs(A[:, None, :] - B[None, :, :])
    if p == np.inf:
        return np.max(diff, axis=2)
    return np.sum(diff ** p, axis=2) ** (1.0 / p)

def parzen_margins(D, y_ref, y_query, k, n_valid, kernel=gausen_kernel):
    kk = min(k, n_valid - 1)              # индекс соседа: берём k-го соседа, но не выходим за число доступных соседей
    order = np.argsort(D, axis=1)         # для каждого объекта получаем индексы, соседей в порядке возрастания расстояния
    sd = np.take_along_axis(D, order, axis=1) # сортируем расстояния от ближайшего к дальнему
    sl = y_ref[order]                    
    h = sd[:, kk] # ширина окна Парзена
    h = np.where(h == 0, 1e-12, h)        
    W = kernel(sd / h[:, None])    
    W[:, kk:] = 0.0                      
    classes = np.unique(y_ref)             
    G = np.stack([(W * (sl == c)).sum(axis=1) for c in classes],axis=1)  # для каждого объекта считаем суммарный вес соседей каждого класса G имеет размер (объекты × классы)
    own = classes[None, :] == y_query[:, None] # создаём маску: какой столбец G соответствует
    g_own = (G * own).sum(axis=1)          # суммарный вес соседей собственного класса
    g_other = np.where(own, -np.inf, G).max(axis=1) # максимальный суммарный вес среди всех остальных классов
    return g_own - g_other

def stolp(X, y, k, delta=0, l0=10, p=2, kernel=gausen_kernel):
    n = len(X)
    classes = np.unique(y)
    D = pairwise_dist(X, X, p)
 
    D_loo = D.copy()
    np.fill_diagonal(D_loo, np.inf)
    margins = parzen_margins(D_loo, y, y, k, n_valid=n - 1, kernel=kernel)
    noise = np.where(margins < delta)[0]
    keep = np.where(margins >= delta)[0]
 
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
        E = rest[m < 0]
        history.append(len(E))
        if len(E) < l0:
            break
        omega.append(E[np.argmin(m[m < 0])])
 
    return np.array(omega), noise, history
# 7. подготовить визуализацию результатов работы алгоритма отбора эталонов;
# 8. сравнить качество работы KNN с и без отбора эталонов; 
# 7. визуализация отбора эталонов

if __name__ == "__main__":
    k_values = list(range(1, 50))
    optimal_k, losses = best_k(X_train, y_train, k_values, p=2, kernel=gausen_kernel)
    print(f"Лучшее k по LOO: {optimal_k}, LOO-ошибка: {losses[k_values.index(optimal_k)]:.4f}")
 
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
    plt.xticks(k_values[::2], fontsize=8) 
    plt.xlabel("число соседей k")
    plt.ylabel("частота ошибок")
    plt.title("Зависимость ошибки от числа соседей")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "loo_vs_k.png", dpi=150, bbox_inches="tight")  
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

    omega, noise, history = stolp(X_train, y_train, optimal_k, delta=0, l0=10)
    X_proto, y_proto = X_train[omega], y_train[omega]
    print(f"\nSTOLP: эталонов {len(omega)} из {len(X_train)}, выбросов {len(noise)}, "
          f"ошибок вне эталонов на последнем шаге {history[-1]}")


    k_proto = min(optimal_k, len(omega) - 1)


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
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), layout="constrained")
    draw_decision_map(axes[0], X_train, y_train, pca, lims, optimal_k, X2, y_train,
                      f"Без отбора: {len(X_train)} объектов, k = {optimal_k}", pt_size=14)
 
    is_proto = np.zeros(len(y_train), dtype=bool)
    is_proto[omega] = True
    is_noise = np.zeros(len(y_train), dtype=bool)
    is_noise[noise] = True
    rest = ~is_proto & ~is_noise
    ax = axes[1]
   
    for c, color, name in ((1, "lime", "класс 1"), (0, "red", "класс 0")):
        m = rest & (y_train == c)
        ax.scatter(X2[m, 0], X2[m, 1], c=color, s=14, alpha=0.35, label=f"остальные, {name}")
    ax.scatter(X2[is_noise, 0], X2[is_noise, 1], marker="x", c="black", s=40,
               label=f"выбросы ({is_noise.sum()})")
    for c, color, name in ((1, "lime", "класс 1"), (0, "red", "класс 0")):
        m = is_proto & (y_train == c)
        ax.scatter(X2[m, 0], X2[m, 1], c=color, edgecolors="black", linewidths=1.2, s=70,
                   label=f"эталоны, {name} ({m.sum()})")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Результат отбора эталонов (STOLP)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
 
    draw_decision_map(axes[2], X_proto, y_proto, pca, lims, k_proto, X2[omega], y_proto,
                      f"С отбором: {len(omega)} эталонов, k = {k_proto}", pt_size=40)
    fig.savefig(OUT_DIR / "stolp_result.png", dpi=150, bbox_inches="tight")  
    plt.show()

    overall = data[target].mean()

    fig, (ax_cat, ax_pc) = plt.subplots(1, 2, figsize=(18, 7), layout="constrained", width_ratios=[1.3, 1])
    labels, shares, counts = [], [], []
    for col in ["ST_Slope", "ExerciseAngina", "ChestPainType", "Sex", "FastingBS", "RestingECG"]:
        g = data.groupby(col)[target].agg(["mean", "count"])
        for level, row in g.iterrows():
            labels.append(f"{col} = {level}")
            shares.append(row["mean"])
            counts.append(int(row["count"]))
    shares = np.array(shares)
    pos = np.arange(len(labels))[::-1]
    ax_cat.barh(pos, shares, color=np.where(shares > overall, "tab:green", "tab:red"))
    ax_cat.axvline(overall, color="black", linestyle="--")
    for y_pos, sh, cnt in zip(pos, shares, counts):
        ax_cat.text(sh + 0.01, y_pos, f"{sh:.0%} (n={cnt})", va="center", fontsize=8)
    ax_cat.set_yticks(pos, labels, fontsize=8)
    ax_cat.set_xlim(0, 1.1)
    ax_cat.set_xlabel("доля пациентов с болезнью сердца в группе")
    ax_cat.set_title("(а) Доля больных по категориальным признакам")
    ax_cat.legend(handles=[Patch(color="tab:green", label="больных больше, чем в среднем"),
                           Patch(color="tab:red", label="больных меньше, чем в среднем"),
                           plt.Line2D([], [], color="black", linestyle="--",
                                      label=f"доля больных во всей выборке ({overall:.0%})")],
                  loc="lower right", fontsize=8)

    feature_names = X.columns.to_numpy()
    pc1 = pca.components_[0]
    order = np.argsort(pc1)
    corr = np.corrcoef(pca.transform(X_train)[:, 0], y_train)[0, 1]
    ax_pc.barh(np.arange(len(order)), pc1[order], color=np.where(pc1[order] < 0, "tab:green", "tab:red"))
    ax_pc.axvline(0, color="black", linewidth=0.8)
    ax_pc.set_yticks(np.arange(len(order)), feature_names[order], fontsize=8)
    ax_pc.set_xlabel("коэффициент признака в PC1")
    ax_pc.set_title(f"(б) Из чего состоит ось PC1 на картах\n(корреляция PC1 с диагнозом {corr:.2f})")
    ax_pc.legend(handles=[Patch(color="tab:green", label="сдвигает объект влево, к больным"),
                          Patch(color="tab:red", label="сдвигает объект вправо, к здоровым")],
                 loc="lower right", fontsize=8)
    fig.savefig(OUT_DIR / "data_features.png", dpi=150, bbox_inches="tight")
    plt.show()

    # рисунок 2: числовые признаки у больных и здоровых
    num_cols = ["Age", "MaxHR", "Oldpeak", "RestingBP", "Cholesterol"]
    fig, axes_num = plt.subplots(1, len(num_cols), figsize=(20, 5.5), layout="constrained")
    for ax, col in zip(axes_num, num_cols):
        values = data[col]
        title = col
        if col == "Cholesterol":                     # нули - скрытые пропуски, на графике не учитываются
            values = values.where(values > 0)
            title = "Cholesterol (без нулевых значений)"
        parts = [values[data[target] == c].dropna() for c in (0, 1)]
        bp = ax.boxplot(parts, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], ("tab:red", "tab:green")):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2], ["нет болезни", "болезнь"])
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.legend(bp["boxes"], [f"нет болезни: медиана {parts[0].median():g}",
                                f"болезнь: медиана {parts[1].median():g}"],
                  fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.suptitle("Числовые признаки у больных и здоровых (ящик - 50% пациентов, линия - медиана)")
    fig.savefig(OUT_DIR / "data_numeric.png", dpi=150, bbox_inches="tight")
    plt.show()

    cm = np.array([[np.sum((y_test == t) & (y_pred_custom == p_)) for p_ in (0, 1)] for t in (0, 1)])
    sensitivity = cm[1, 1] / cm[1].sum()
    specificity = cm[0, 0] / cm[0].sum()
    fig, ax_cm = plt.subplots(figsize=(7, 5.5), layout="constrained")
    im = ax_cm.imshow(cm, cmap="Blues")
    names = [["верно: здоров", "ложная тревога"], ["пропуск болезни", "верно: болен"]]
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=10,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax_cm.set_xticks([0, 1], ["нет болезни", "болезнь"])
    ax_cm.set_yticks([0, 1], ["нет болезни", "болезнь"])
    ax_cm.set_xlabel("предсказание модели")
    ax_cm.set_ylabel("истинный диагноз")
    ax_cm.set_title(f"Матрица ошибок окна Парзена на тесте (k = {optimal_k})\n"
                    f"чувствительность {sensitivity:.0%}, специфичность {specificity:.0%}")
    cbar = plt.colorbar(im, ax=ax_cm)
    cbar.set_label("число пациентов")
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nЧувствительность (найдено больных): {cm[1, 1]} из {cm[1].sum()} = {sensitivity:.2%}")
    print(f"Специфичность (верно определено здоровых): {cm[0, 0]} из {cm[0].sum()} = {specificity:.2%}")
    print(f"Пациентов с Cholesterol = 0: {(data['Cholesterol'] == 0).sum()}, "
          f"из них больны {data.loc[data['Cholesterol'] == 0, target].mean():.0%}")
