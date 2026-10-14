import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import data
from knn import condensed_nn, knn_predict

X_synthetic = np.array([[0.0],[0.1],[0.2],[5.0],[5.1],[5.2]])
y_synthetic = np.array([0,0,0,1,1,1])
U_synthetic,y_synthetic_selected = condensed_nn(X_synthetic,y_synthetic,random_state=42)
synthetic_errors = 0
for i in range(len(y_synthetic)):
    pred = knn_predict(U_synthetic,y_synthetic_selected,X_synthetic[i],k=1)
    synthetic_errors += int(pred != y_synthetic[i])
print(
    "синтетический пример: эталонов",len(y_synthetic_selected),
    "из",len(y_synthetic),"ошибок 1-NN",synthetic_errors,
)
if synthetic_errors != 0 or len(y_synthetic_selected) != 2:
    raise AssertionError(
        f"на синтетическом примере ждали 2 эталона и 0 ошибок, "
        f"получили {len(y_synthetic_selected)}, {synthetic_errors}"
    )


started = time.perf_counter()
U,U_labels = condensed_nn(data.X_train,data.y_train,random_state=42)
select_ms = (time.perf_counter() - started) * 1000

train_wrong = 0
for i in range(len(data.y_train)):
    pred = knn_predict(U,U_labels,data.X_train[i],k=1)
    train_wrong += int(pred != data.y_train[i])
print("train: эталонов",len(U_labels),"из",len(data.y_train))
print("ошибок 1-NN по эталонам на train:",train_wrong)
print("отбор занял, ms:",round(select_ms,1))
for cls,name in enumerate(data.label_encoder.classes_):
    print(name,"эталонов",int((U_labels == cls).sum()))
if train_wrong != 0:
    raise AssertionError("после отбора 1-NN на эталонах обязан верно классифицировать весь train")

KNN_K = 13
k_on_u = min(KNN_K,len(U_labels))


def predict_ms(X_fit,y_fit,k,repeats=3):
    spent = []
    pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = np.array([knn_predict(X_fit,y_fit,x,k) for x in data.X_test])
        spent.append((time.perf_counter() - t0) * 1000)
    return pred, min(spent)


full_pred,full_ms = predict_ms(data.X_train,data.y_train,KNN_K)
full_1_pred,full_1_ms = predict_ms(data.X_train,data.y_train,1)
u_k_pred,u_k_ms = predict_ms(U,U_labels,k_on_u)
u_1_pred,u_1_ms = predict_ms(U,U_labels,1)

full_acc = accuracy_score(data.y_test,full_pred)
full_1_acc = accuracy_score(data.y_test,full_1_pred)
u_k_acc = accuracy_score(data.y_test,u_k_pred)
u_1_acc = accuracy_score(data.y_test,u_1_pred)

print()
print(f"{'метрика':<28}{'весь train':>14}{'эталоны':>14}")
print(f"{'Размер выборки':<28}{len(data.y_train):14d}{len(U_labels):14d}")
print(f"{'Время отбора (ms)':<28}{0:14.1f}{select_ms:14.1f}")
print(f"{'k':<28}{KNN_K:14d}{k_on_u:14d}")
print(f"{'Accuracy при своём k':<28}{full_acc:14.3f}{u_k_acc:14.3f}")
print(f"{'Время предсказания (ms)':<28}{full_ms:14.1f}{u_k_ms:14.1f}")
print(f"{'Accuracy 1-NN':<28}{full_1_acc:14.3f}{u_1_acc:14.3f}")
print(f"{'Время 1-NN (ms)':<28}{full_1_ms:14.1f}{u_1_ms:14.1f}")

names = list(data.label_encoder.classes_)
X_mm = data.scaler.inverse_transform(data.X_train)
U_mm = data.scaler.inverse_transform(U)

fig,axes = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
panels = (
    (axes[0],X_mm,data.y_train,"o",20,0.45,f"Train ({len(data.y_train)})"),
    (axes[1],U_mm,U_labels,"x",70,1.0,f"Эталоны ({len(U_labels)} из {len(data.y_train)})"),
)
for ax,coords,labels,marker,size,alpha,title in panels:
    for cls,name in enumerate(names):
        mask = labels == cls
        ax.scatter(coords[mask,0],coords[mask,1],label=name,marker=marker,s=size,alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("длина клюва, мм")
    ax.set_ylabel("глубина клюва, мм")
    ax.legend()
    ax.grid(True,alpha=0.3)
fig.tight_layout()
fig.savefig("plots/prototypes.png",format="png")
plt.close()
print("график сохранён в plots/prototypes.png")
