import numpy as np

from kernels import GaussianKernel
from metric_classifier import ParzenClassifier


def loo(x,y, k_n):
    emp_risk = []

    ker = GaussianKernel()
    min_k = 999999999999
    e_min = 10**8
    n_iter = x.shape[0]
    clf = ParzenClassifier()
    for k in range(1, k_n):
        print(k)
        e = 0
        for i in range(n_iter):
            x_mod = np.delete(x, i, axis = 0)
            y_mod = np.delete(y,i, axis = 0)
            clf.fit(x_mod, y_mod)
            pred = clf.predict(x[i,:].reshape(1,-1), ker, k)
            e+= y[i].item() != pred.item()
        if e < e_min:

            e_min = e
            min_k = k
        emp_risk.append(e/x.shape[0])
    return min_k, e_min, emp_risk



