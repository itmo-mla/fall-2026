import matplotlib.pyplot as plt
import numpy as np
import data
from knn import knn_predict, loo_cv, parzen_predict

X_synthetic = np.array([[0.0],[1.0],[1.1]])
y_synthetic = np.array([0,1,1])
synthetic_risk = loo_cv(X_synthetic,y_synthetic,k=1,model_func=knn_predict)
print("синтетический пример, LOO k=1, риск:", round(synthetic_risk,3), "ждали 0.333")
if abs(synthetic_risk - 1/3) > 1e-9:
    raise AssertionError(f"LOO на синтетическом примере дал {synthetic_risk}, ждали 1/3")

k_values = list(range(1,21))
knn_errors = []
parzen_errors = []
print("=== LOO по train, n =", len(data.y_train), "===")
for k in k_values:
    knn_risk = loo_cv(data.X_train,data.y_train,k,knn_predict)
    parzen_risk = loo_cv(data.X_train,data.y_train,k,parzen_predict)
    knn_errors.append(knn_risk)
    parzen_errors.append(parzen_risk)
    print(f"k={k}: knn {knn_risk:.4f}, parzen {parzen_risk:.4f}", flush=True)

best_knn_k = k_values[int(np.argmin(knn_errors))]
best_parzen_k = k_values[int(np.argmin(parzen_errors))]
knn_tied = [k for k,err in zip(k_values,knn_errors) if abs(err - min(knn_errors)) < 1e-12]
parzen_tied = [k for k,err in zip(k_values,parzen_errors) if abs(err - min(parzen_errors)) < 1e-12]
print("оптимальное k для KNN:", best_knn_k, "риск", round(min(knn_errors),4), "ничья k:", knn_tied)
print("оптимальное k для Парзена:", best_parzen_k, "риск", round(min(parzen_errors),4), "ничья k:", parzen_tied)

plt.figure()
plt.plot(k_values,knn_errors,marker="o",label="KNN")
plt.plot(k_values,parzen_errors,marker="o",label="Парзен")
plt.xlabel("k")
plt.ylabel("LOO error rate")
plt.title("Подбор параметра k")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("plots/loo_error.png", format="png")
plt.close()
print("график сохранён в plots/loo_error.png")
