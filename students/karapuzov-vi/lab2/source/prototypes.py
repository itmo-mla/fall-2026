import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import data
from knn import ccv_risk, knn_predict, select_prototypes_ccv

CONTROL_LENGTH = 1

X_synthetic = np.array([[0.0],[0.1],[0.2],[5.0],[5.1],[5.2]])
y_synthetic = np.array([0,0,0,1,1,1])
idx_synthetic,history_synthetic,dist_synthetic = select_prototypes_ccv(
    X_synthetic,y_synthetic,control_length=CONTROL_LENGTH,
)
q_synthetic,pi_synthetic = ccv_risk(
    dist_synthetic,y_synthetic,idx_synthetic,control_length=CONTROL_LENGTH,
)
print(
    "синтетический пример: эталонов",len(idx_synthetic),
    "из",len(y_synthetic),
    "Q=Π(1) =",round(q_synthetic,3),
)
if abs(q_synthetic - history_synthetic[-1]) > 1e-9 or q_synthetic != 0.0:
    raise AssertionError(f"на синтетическом примере ждали Q=0, получили {q_synthetic}")
for cls in (0,1):
    if int(np.sum(y_synthetic[idx_synthetic] == cls)) < 2:
        raise AssertionError("в каждом классе нужно минимум два эталона: сосед эталона не он сам")

started = time.perf_counter()
idx,history,dist = select_prototypes_ccv(
    data.X_train,data.y_train,control_length=CONTROL_LENGTH,
)
select_ms = (time.perf_counter() - started) * 1000
U = data.X_train[idx]
U_labels = data.y_train[idx]

q_omega,pi_omega = ccv_risk(dist,data.y_train,idx,control_length=CONTROL_LENGTH)
q_full,pi_full = ccv_risk(
    dist,data.y_train,np.arange(len(data.y_train)),control_length=CONTROL_LENGTH,
)
dist_self = dist.copy()
np.fill_diagonal(dist_self,0.0)
q_with_self,_ = ccv_risk(dist_self,data.y_train,idx,control_length=CONTROL_LENGTH)

print("длина контроля L =",CONTROL_LENGTH,", R = (1,), Q = Π(1,Ω)")
print("train: эталонов",len(idx),"из",len(data.y_train))
print("Π(1) на всём train, самоисключение:",round(q_full,4))
print("Π(1) на эталонах, самоисключение:",round(q_omega,4))
print("Π(1) на эталонах, если объект сосед сам себе:",round(max(q_with_self,0.0),4))
print("шагов наращивания:",len(history) - 1, "отбор, ms:",round(select_ms,1))
for cls,name in enumerate(data.label_encoder.classes_):
    print(name,"эталонов",int((U_labels == cls).sum()))
if abs(q_omega - history[-1]) > 1e-9:
    raise AssertionError("критерий после отбора не совпал с историей жадного шага")


def predict_ms(X_fit,y_fit,k,repeats=3):
    spent = []
    pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = np.array([knn_predict(X_fit,y_fit,x,k) for x in data.X_test])
        spent.append((time.perf_counter() - t0) * 1000)
    return pred, min(spent)


full_pred,full_ms = predict_ms(data.X_train,data.y_train,1)
u_pred,u_ms = predict_ms(U,U_labels,1)
full_acc = accuracy_score(data.y_test,full_pred)
u_acc = accuracy_score(data.y_test,u_pred)

print()
print(f"{'метрика':<42}{'весь train':>14}{'эталоны CCV':>14}")
print(f"{'Размер выборки':<42}{len(data.y_train):14d}{len(U_labels):14d}")
print(f"{'Π(1), самоисключение':<42}{q_full:14.4f}{q_omega:14.4f}")
print(f"{'Accuracy 1-NN на test':<42}{full_acc:14.3f}{u_acc:14.3f}")
print(f"{'Время предсказания (ms)':<42}{full_ms:14.1f}{u_ms:14.1f}")
print(f"{'Время отбора (ms)':<42}{0:14.1f}{select_ms:14.1f}")

names = list(data.label_encoder.classes_)
X_mm = data.scaler.inverse_transform(data.X_train)
U_mm = data.scaler.inverse_transform(U)

fig,axes = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
panels = (
    (axes[0],X_mm,data.y_train,"o",20,0.45,f"Train ({len(data.y_train)})"),
    (axes[1],U_mm,U_labels,"x",70,1.0,f"Эталоны CCV ({len(U_labels)} из {len(data.y_train)})"),
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
