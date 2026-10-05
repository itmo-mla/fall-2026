# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
# 2. реализовать алгоритм KNN с методом окна Парзена переменной ширины;
#    1. в качестве ядра можно использовать гауссово ядро;
# 3. подобрать параметр k методом скользящего контроля (LOO);
# 4. обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;
# 5. сравнить с [эталонной](https://scikit-learn.org/stable/) реализацией KNN;
#    1. сравнить качество работы алгоритмов;
# 6. реализовать алгоритм отбора эталонов;
# 7. подготовить визуализацию результатов работы алгоритма отбора эталонов;
# 8. сравнить качество работы KNN с и без отбора эталонов; 
# 9. подготовить небольшой отчет о проделанной работе.

import os
import numpy as np
import pandas as pd
import kagglehub
import knn
import plots
import comparings
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def path(filename):
    return os.path.join(OUTPUT_DIR, filename)

# 1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
                                                           
def load_data():
    df = pd.read_csv(os.path.join(kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data"), "data.csv"))
    y = (df["diagnosis"] == "M").astype(int).values              # 1 - злокачественная опухоль
    X = df.drop(columns=["id", "diagnosis", "Unnamed: 32"]).values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def sklearn_predict(X_train, y_train, X_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def parzen_classifier(X_fit, y_fit, Y, k):
    # cdist при евклидовой метрике = rho(), но быстрее: для сетки нужны десятки тысяч точек
    return lambda X: knn.predict(cdist(X, X_fit), y_fit, Y, k, knn.gaussian_kernel)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    D_train = knn.distance_matrix(X_train, knn.rho)             # train × train, для LOO
    D_test = knn.distance_matrix(X_test, knn.rho, Z=X_train)    # test × train, для предсказаний

    Y = np.unique(y_train)

    ks = range(1, 51)
    risks = knn.loo_curve(D_train, y_train, Y, ks, knn.gaussian_kernel)

    best_k = ks[int(np.argmin(risks))]
    y_pred = knn.predict(D_test, y_train, Y, best_k, knn.gaussian_kernel)
    y_sk = sklearn_predict(X_train, y_train, X_test, best_k)

    print("self KNN : accuracy =", accuracy_score(y_test, y_pred), " F1 =", f1_score(y_test, y_pred))
    print("sklearn KNN: accuracy =", accuracy_score(y_test, y_sk), " F1 =", f1_score(y_test, y_sk))

    pca = PCA(n_components=2).fit(X_train)
    Z_test = pca.transform(X_test)

    plots.plot_pca_predictions(Z_test, y_test, y_pred, path("pca_parzen.png"), "Parzen-KNN",
                               pca, parzen_classifier(X_train, y_train, Y, best_k))
    plots.plot_pca_predictions(Z_test, y_test, y_sk, path("pca_sklearn.png"), "sklearn KNN",
                               pca, lambda X: sklearn_predict(X_train, y_train, X, best_k))

    plots.plot_loo_risk(ks, risks, best_k, path("loo_risk.png"))

    # 6, 8. отбор эталонов и сравнение KNN с отбором и без
    res = comparings.compare_with_prototypes(D_train, D_test, y_train, y_test, ks, best_k, k_ctrl=1)

    # 7. визуализация отбора эталонов
    plots.plot_ccv_history(res["history"], path("ccv_history.png"))
    plots.plot_prototypes(pca.transform(X_train), y_train, res["omega"], path("prototypes.png"))

    omega, k_omega = res["omega"], res["best_k_omega"]
    y_pred_omega = knn.predict(D_test[:, omega], y_train[omega], Y, k_omega, knn.gaussian_kernel)
    plots.plot_pca_predictions(Z_test, y_test, y_pred_omega, path("pca_parzen_prototypes.png"), "Parzen-KNN с эталонами",
                               pca, parzen_classifier(X_train[omega], y_train[omega], Y, k_omega))
