import numpy as np


def calculate_pca(X_train):
    v, d, u_t = np.linalg.svd(X_train, full_matrices=False)
    lam = d**2
    u = u_t.T
    return lam, u


def calculate_g(X_train, u, m):
    # g - это f на u
    u_m = u[:, :m]
    g = X_train @ u_m
    return g


def calculate_f_hat(g, u, m):
    # f_hat - это g на u.T
    u_m = u[:, :m]
    f_hat = g @ u_m.T
    return f_hat


def calculate_loss(f, f_hat):
    diff = (f_hat - f)**2
    return np.sum(diff)


def calculate_e_m(lam):
    e_m = []
    denominator = np.sum(lam)
    for i in range(len(lam)+1):
        e_m.append(np.sum(lam[i:])/denominator)
    return np.array(e_m)


def calculate_m_eps(e_m, eps):
    if eps < 0:
        raise ValueError('eps cannot be negative')
    mask = e_m <= eps
    return np.argmax(mask)


def calculate_m_slope(e_m):
    numerator = e_m[:-2]
    denominator = e_m[1:-1]
    ratios = numerator / denominator
    return np.argmax(ratios)+1


def calculate_w(g, lam, y_c):
    w_list = []
    for i in range(g.shape[1]):
        w_list.append(np.sum(g[:,i] * y_c)/lam[i])
    return w_list


def forecast(g, w, y_mean):
    return g @ w + y_mean
