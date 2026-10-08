from initialization import calculate_w_correlation, calculate_w_random, calculate_q_start
from metric import calculate_accuracy
from model import calculate_sgd


# 9.1
def run_correlation_init(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, learning_rate):
    w_start = calculate_w_correlation(X_train, y_train)
    Q_start = calculate_q_start(X_train, y_train, w_start, q_start_size)
    w_final, Q_final, count_final = calculate_sgd(X_train, y_train, w_start, momentum, q_update_rate, Q_start, l2_strength, margin_epsilon, tolerance, learning_rate, sampling="random")
    return {"w_start": w_start, "w": w_final, "Q": Q_final, "iters": count_final}

# 9.2
def run_multistart(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, n_starts, learning_rate):
    acc_train_best = 0
    best = None
    all_acc_train = []

    for i in range(n_starts):
        n = X_train.shape[1]
        w_start = calculate_w_random(n)
        Q_start = calculate_q_start(X_train, y_train, w_start, q_start_size)
        w_final, Q_final, count_final = calculate_sgd(X_train, y_train, w_start, momentum, q_update_rate, Q_start, l2_strength, margin_epsilon, tolerance, learning_rate, sampling="random")
        acc_train = calculate_accuracy(X_train, y_train, w_final)
        all_acc_train.append(acc_train)
        if acc_train > acc_train_best:
            acc_train_best = acc_train
            best = {"w_start": w_start, "w": w_final, "Q": Q_final, "iters": count_final}

    best["all_acc_train"] = all_acc_train
    return best

# 9.3
def run_sampling_comparison(X_train, y_train, momentum, q_update_rate, l2_strength, margin_epsilon, tolerance, q_start_size, w_start, learning_rate):
    Q_start = calculate_q_start(X_train, y_train, w_start, q_start_size)

    w_margin, Q_margin, count_margin = calculate_sgd(X_train, y_train, w_start.copy(), momentum, q_update_rate, Q_start, l2_strength, margin_epsilon, tolerance, learning_rate,  sampling="margin")
    w_uni, Q_uni, count_uni = calculate_sgd(X_train, y_train, w_start.copy(), momentum, q_update_rate, Q_start, l2_strength, margin_epsilon, tolerance, learning_rate, sampling="random")

    margin = {"w_start": w_start, "w": w_margin, "Q": Q_margin, "iters": count_margin}
    uniform = {"w_start": w_start, "w": w_uni, "Q": Q_uni, "iters": count_uni}
    return margin, uniform
