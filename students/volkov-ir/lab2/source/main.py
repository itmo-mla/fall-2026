import numpy as np
from data_preprocessing import prepare_data

# X уже стандартизованы, y — номера видов 0/1/2
X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data()


def calculate_knn(X, y, u, k):
    #эвклидово расстояние
    square = np.square(u-X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    shear = nearest_idx[0:k]
    shear_y = y[shear]
    classes, counts = np.unique(shear_y, return_counts=True)
    index_max = np.argmax(counts)
    return classes[index_max]


def calculate_parzen(X, y, u, k):
    square = np.square(u - X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    h = dist_ux[nearest_idx[k]]
    shear = nearest_idx[0:k]
    shear_y = y[shear]
    p = dist_ux[shear]
    w = np.exp(-(p**2/(2*h**2)))
    sums_per_class = np.bincount(shear_y, weights=w)
    pred = np.argmax(sums_per_class)
    return pred

def calculate_parzen_lecture(X, y, u, k):
    square = np.square(u - X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    h = dist_ux[nearest_idx[k]]
    p = dist_ux
    w = np.exp(-(p ** 2 / (2 * h ** 2)))
    sums_per_class = np.bincount(y, weights=w)
    pred = np.argmax(sums_per_class)
    return pred

def calculate_loo(X, y, k_values, method):
    loss = []
    for k in k_values:
        count_loss = 0
        for i in range(0, len(X)):
            X_k = X.copy()
            y_k = y.copy()
            u = X_k[i]
            y_i = y_k[i]
            X_k = np.delete(X_k, i , 0)
            y_k = np.delete(y_k, i, 0)
            if method == "knn":
                pred = calculate_knn(X_k, y_k, u, k)
            elif method == "parzen":
                pred = calculate_parzen(X_k, y_k, u, k)
            elif method == "parzen_lecture":
                pred = calculate_parzen_lecture(X_k, y_k, u, k)
            else:
                raise ValueError("method должен быть 'knn', 'parzen' или 'parzen_lecture'")
            if pred != y_i:
                count_loss += 1
        loss.append(count_loss/len(X))
    return loss


X = np.array([[1.0, 1.2], [2.3, 0.9], [1.4, 2.1], [2.6, 2.8],
              [6.1, 5.3], [5.7, 6.4], [7.2, 5.1]])
y = np.array([0, 0, 1, 1, 1, 1, 0])
k_values = [4, 1, 2]
print(calculate_loo(X, y, k_values, method="knn"))
print(calculate_loo(X, y, k_values, method="parzen"))
print(calculate_loo(X, y, k_values, method="parzen_lecture"))

