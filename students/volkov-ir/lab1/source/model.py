import numpy as np

def calculate_margin(X, y, w):
    mult_vector = np.dot(X, w)
    return np.multiply(mult_vector, y)


def calculate_gradient(X, y, margin):
    coef = (-2*(1-margin)*y)
    return (coef.reshape(-1,1)*X).sum(axis=0)


def calculate_loss(margin):
    return (1-margin)**2


def calculate_q(q_update_rate, q_old, loss):
    return q_update_rate*loss+(1-q_update_rate)*q_old


def calculate_sgd(X, y, w, momentum, q_update_rate, Q_old, l2_strength, margin_epsilon, tolerance, learning_rate, sampling):
    count = 0
    limit = 1 * 10**5
    recalc_every = 100
    Q_ep = Q_old
    v = np.zeros_like(w)
    p = None
    stable_count = 0
    while True:
        if sampling == "random":
            i = np.random.choice(len(X))

        elif sampling == "margin":
            if count % recalc_every == 0:
                p = calculate_p(calculate_margin(X, y, w), margin_epsilon)
            i = np.random.choice(len(X), p=p)
        else:
            raise ValueError("sampling должен быть 'random' или 'margin'")

        x_i, y_i = X[i], y[i]
        Margin_i = calculate_margin(x_i, y_i, w)
        loss_i = calculate_loss(Margin_i)
        gradient_i = calculate_gradient(x_i, y_i, Margin_i)
        l2 = calculate_l2(w, l2_strength)
        gradient_l2 = gradient_i+l2
        h_i = calculate_h(Margin_i, x_i, y_i, gradient_l2) * learning_rate
        v = momentum*v + (np.multiply(h_i,gradient_l2))
        w = w-v
        Q = calculate_q(q_update_rate, Q_old, loss_i)
        count += 1
        if count%len(X)==0:
            if abs(Q - Q_ep) <= tolerance:
                stable_count +=1
            else:
                stable_count = 0
            Q_ep = Q
            if stable_count == 3:
                break
        if count >= limit:
            break
        Q_old = Q
    return w, Q, count


def calculate_l2(w, l2_strength):
    l2 = l2_strength * w.copy()
    l2[0] = 0
    return l2


def calculate_h(M, x, y, gradient):
    g = (gradient @ x)*y
    return (M-1)/g


def calculate_p(M, margin_epsilon):
    abs_M = np.abs(M)
    weight = 1/(abs_M+margin_epsilon)
    weight_sum = weight.sum()
    return weight/weight_sum


def calculate_index(X, p):
    index = np.random.choice(len(X), size=1, p=p)
    return index[0]