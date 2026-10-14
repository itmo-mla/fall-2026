import numpy as np
from collections import Counter

def knn_predict(X_train,y_train,x_test,k=1):
    distances = [euclid_distance(x_test,row) for row in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]

#Евклидово расстояние 
def euclid_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2))


#Ядро активация весов
def gausian_kernel(distance,h):
    z = distance/h
    return np.exp(-z**2/2)

#Пархена окно
def parzen_predict(X_train,y_train,x_test,k):
    distances = [euclid_distance(x_test,row) for row in X_train]
    k_indices = np.argsort(distances)[:k]
    h = distances[k_indices[-1]]
    if h == 0:
        labels = [y_train[i] for i,d in enumerate(distances) if d == 0]
        return Counter(labels).most_common()[0][0]
    weights = [gausian_kernel(d,h) for d in distances]
    scores = {}
    for label,weight in zip(y_train,weights):
        scores[label] = scores.get(label,0.0) + weight
    return max(scores,key=scores.get)

#Ошибку считаем
def loo_cv(X,y,k,model_func):
    n = len(y)
    errors = 0
    for i in range(n):
        X_train = np.delete(X,i,axis=0)
        y_train = np.delete(y,i)
        x_test = X[i]
        y_true = y[i]
        y_pred = model_func(X_train,y_train,x_test,k)
        errors += int(y_pred != y_true)
    return errors / n

#Отбор эталонов
def condensed_nn(X,y,max_iter=50,random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    selected = [int(rng.integers(0,n))]
    chosen = set(selected)
    changed = True
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        for i in range(n):
            if i in chosen:
                continue
            y_pred = knn_predict(X[selected],y[selected],X[i],k=1)
            if y_pred != y[i]:
                selected.append(i)
                chosen.add(i)
                changed = True
        iteration += 1
    selected = np.array(selected)
    return X[selected], y[selected]

def check_knn_and_parzen():
    X_synthetic = np.array([[0.0],[1.0],[1.1]])
    y_synthetic = np.array([0,1,1])
    synthetic_cases = [
        (np.array([0.2]),1,0,0),
        (np.array([0.2]),2,0,1),
        (np.array([0.0]),1,0,0),
    ]
    print("=== синтетический пример ===")
    print("точки: класс 0 в 0.0, класс 1 в 1.0 и 1.1")
    for query,k,expected_knn,expected_parzen in synthetic_cases:
        knn_pred = knn_predict(X_synthetic,y_synthetic,query,k)
        parzen_pred = parzen_predict(X_synthetic,y_synthetic,query,k)
        print(
            f"запрос {query.tolist()} k={k}: "
            f"knn {knn_pred} (ждали {expected_knn}), "
            f"parzen {parzen_pred} (ждали {expected_parzen})"
        )
        if knn_pred != expected_knn or parzen_pred != expected_parzen:
            raise AssertionError(
                f"синтетический пример {query.tolist()} k={k}: "
                f"knn {knn_pred}/{expected_knn}, parzen {parzen_pred}/{expected_parzen}"
            )

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import data

    print("=== пингвины: KNN против sklearn, Парзен против своего KNN по accuracy ===")
    for k in (1,5):
        sklearn_knn = KNeighborsClassifier(n_neighbors=k,algorithm="brute",metric="euclidean")
        sklearn_knn.fit(data.X_train,data.y_train)
        sklearn_pred = sklearn_knn.predict(data.X_test)
        knn_pred = np.array([knn_predict(data.X_train,data.y_train,x,k) for x in data.X_test])
        parzen_pred = np.array([parzen_predict(data.X_train,data.y_train,x,k) for x in data.X_test])
        mismatches = int((knn_pred != sklearn_pred).sum())
        print(
            f"k={k}: knn {accuracy_score(data.y_test,knn_pred):.3f}, "
            f"sklearn {accuracy_score(data.y_test,sklearn_pred):.3f}, "
            f"расхождений knn/sklearn {mismatches}, "
            f"parzen {accuracy_score(data.y_test,parzen_pred):.3f}"
        )
        if k == 1 and mismatches != 0:
            raise AssertionError("при k=1 ничьей нет, KNN должен совпасть со sklearn")

if __name__ == "__main__":
    check_knn_and_parzen()
