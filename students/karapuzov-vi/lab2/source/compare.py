import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

import data
from knn import knn_predict, parzen_predict
KNN_K = 13
PARZEN_K = 3
CLASS_NAMES = list(data.label_encoder.classes_)


def predict_ours(model_func,k):
    return np.array([model_func(data.X_train,data.y_train,x,k) for x in data.X_test])


def best_ms(fn,repeats):
    spent = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        spent.append((time.perf_counter() - started) * 1000)
    return result, min(spent)


def class_scores(y_true,y_pred):
    precision,recall,f1,_ = precision_recall_fscore_support(
        y_true,y_pred,labels=[0,1,2],zero_division=0
    )
    return precision,recall,f1


sklearn_knn = KNeighborsClassifier(n_neighbors=KNN_K,algorithm="brute",metric="euclidean")
sklearn_knn.fit(data.X_train,data.y_train)

our_pred,our_ms = best_ms(lambda: predict_ours(knn_predict,KNN_K),repeats=3)
sklearn_pred,sklearn_ms = best_ms(lambda: sklearn_knn.predict(data.X_test),repeats=30)
parzen_pred,parzen_ms = best_ms(lambda: predict_ours(parzen_predict,PARZEN_K),repeats=3)

our_acc = accuracy_score(data.y_test,our_pred)
sklearn_acc = accuracy_score(data.y_test,sklearn_pred)
parzen_acc = accuracy_score(data.y_test,parzen_pred)
mismatches = int((our_pred != sklearn_pred).sum())

our_p,our_r,our_f1 = class_scores(data.y_test,our_pred)
sk_p,sk_r,sk_f1 = class_scores(data.y_test,sklearn_pred)
pz_p,pz_r,pz_f1 = class_scores(data.y_test,parzen_pred)

print("=== test, объекты",len(data.y_test),"===")
print(f"KNN k={KNN_K}, Парзен k={PARZEN_K}")
print("расхождений меток KNN и sklearn:",mismatches)
print("разница accuracy KNN и sklearn:",round(abs(our_acc - sklearn_acc),4))
print()
header = f"{'метрика':<22}{'KNN':>10}{'sklearn':>10}{'Парзен':>10}"
print(header)
print(f"{'Accuracy':<22}{our_acc:10.3f}{sklearn_acc:10.3f}{parzen_acc:10.3f}")
print(f"{'Время (ms)':<22}{our_ms:10.1f}{sklearn_ms:10.1f}{parzen_ms:10.1f}")
for idx,name in enumerate(CLASS_NAMES):
    print(f"{'Precision ' + name:<22}{our_p[idx]:10.3f}{sk_p[idx]:10.3f}{pz_p[idx]:10.3f}")
    print(f"{'Recall ' + name:<22}{our_r[idx]:10.3f}{sk_r[idx]:10.3f}{pz_r[idx]:10.3f}")
    print(f"{'F1 ' + name:<22}{our_f1[idx]:10.3f}{sk_f1[idx]:10.3f}{pz_f1[idx]:10.3f}")

if abs(our_acc - sklearn_acc) >= 0.02:
    raise AssertionError("accuracy KNN и sklearn разъехались больше чем на 2%")
