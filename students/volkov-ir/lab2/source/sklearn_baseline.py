import time
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from loo_experiment import select_k
from main import X_test, X_train, calculate_knn, calculate_loo, calculate_parzen, calculate_parzen_lecture, class_names, y_test, y_train

K_CHECK = list(range(1, 51))


def gaussian_parzen_weights(dist):
    # sklearn отдаёт расстояния до k+1 соседей: последний задаёт ширину окна и сам не голосует
    h = dist[:, -1:]
    w = np.exp(-dist ** 2 / (2 * h ** 2))
    w[:, -1] = 0
    return w


def gaussian_parzen_lecture_weights(dist, k):
    # sklearn отдаёт расстояния до всех объектов по возрастанию: ширина окна — (k+1)-й из них
    h = dist[:, k:k + 1]
    return np.exp(-dist ** 2 / (2 * h ** 2))


# у модели с гауссовыми весами n_neighbors = k+1; у варианта лекции соседи — вся обучающая выборка (n объектов)
SKLEARN_MODELS = {
    "sklearn KNN uniform": lambda k, n: KNeighborsClassifier(n_neighbors=k),
    "sklearn KNN distance": lambda k, n: KNeighborsClassifier(n_neighbors=k, weights="distance"),
    "sklearn KNN + гауссово окно": lambda k, n: KNeighborsClassifier(n_neighbors=k + 1, weights=gaussian_parzen_weights),
    "sklearn гауссово окно (лекция)": lambda k, n: KNeighborsClassifier(n_neighbors=n, weights=partial(gaussian_parzen_lecture_weights, k=k)),
}
OWN_MODELS = {"свой KNN": ("knn", calculate_knn), "своё окно Парзена (по k)": ("parzen", calculate_parzen), "своё окно Парзена (лекция)": ("parzen_lecture", calculate_parzen_lecture)}


def sklearn_loo(make_model):
    # в LOO обучающая часть — ℓ-1 объект
    return np.array([1 - cross_val_score(make_model(k, len(X_train) - 1), X_train, y_train, cv=LeaveOneOut(), n_jobs=-1).mean() for k in K_CHECK])


def evaluate(name, k, predict):
    start = time.perf_counter()
    pred = predict()
    elapsed = (time.perf_counter() - start) * 1000
    return {"name": name, "k": k, "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred, average="macro"), "errors": int((pred != y_test).sum()), "ms": elapsed, "pred": pred}


if __name__ == "__main__":
    own_loo = {name: np.array(calculate_loo(X_train, y_train, K_CHECK, method)) for name, (method, _) in OWN_MODELS.items()}
    sk_loo = {name: sklearn_loo(make) for name, make in SKLEARN_MODELS.items()}

    # проверка реализаций: при тех же правилах LOO своих функций и sklearn должен совпасть при каждом k
    pairs = [("свой KNN", "sklearn KNN uniform"), ("своё окно Парзена (по k)", "sklearn KNN + гауссово окно"), ("своё окно Парзена (лекция)", "sklearn гауссово окно (лекция)")]
    for own, sk in pairs:
        print(f"LOO «{own}» = «{sk}» при {np.isclose(own_loo[own], sk_loo[sk]).sum()} из {len(K_CHECK)} k")

    # k выбирается одинаково для всех: минимум LOO на train, при равенстве — наибольшее k
    k_best = {name: select_k(K_CHECK, loss) for name, loss in {**own_loo, **sk_loo}.items()}

    # финальная оценка — один раз на отложенном test
    rows = [evaluate(name, k_best[name], lambda f=func, k=k_best[name]: np.array([f(X_train, y_train, u, k) for u in X_test])) for name, (_, func) in OWN_MODELS.items()]
    for name, make in SKLEARN_MODELS.items():
        model = make(k_best[name], len(X_train)).fit(X_train, y_train)
        rows.append(evaluate(name, k_best[name], lambda m=model: m.predict(X_test)))

    print(f"\n{'модель':<32}{'k':>4}{'accuracy':>10}{'macro-F1':>10}{'ошибок':>8}")
    for r in rows:
        print(f"{r['name']:<32}{r['k']:>4}{r['accuracy']:>10.3f}{r['f1']:>10.3f}{r['errors']:>8}")

    print(f"\nматрицы ошибок (строки — истинный вид, столбцы — предсказанный, порядок {class_names}):")
    for r in rows:
        print(f"{r['name']:<32}", confusion_matrix(y_test, r["pred"]).tolist())
