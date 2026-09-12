from lin_clf.data.utils import fetch_batch, load_data, preprocess
from lin_clf.model import LinearClassifier, Model
from lin_clf.model.WeightInit import init_correlation, init_random
from lin_clf.optimizer import SGD
from lin_clf.utils import linear_ab
from lin_clf.visualization import MetricTracker


def _train(
    X_train,
    X_test,
    y_train,
    y_test,
    lin_model: Model,
    epochs=10_000,
    h=0.001,
    momentum_k=0.01,
    batch_size=2**6,
    visual_smooth=0.01,
    train_name="",
    fetch_prob=False,
    draw_loss=True,
):
    optimizer = SGD(lin_model, momentum_k, h)

    tracker = MetricTracker()
    q = None
    q_list = list()
    prob = None
    for i_epoch in range(epochs):
        if fetch_prob:
            margin = lin_model.predict_margin(lin_model._add_ones(X_train), y_train)
            prob = (1 - margin) ** 2
            prob = (prob / prob.sum())[:, 0]
        mini_batch_x, mini_batch_y = fetch_batch(X_train, y_train, batch_size, prob)

        optimizer.step(mini_batch_x, mini_batch_y)

        loss = float(lin_model.loss(mini_batch_x, mini_batch_y))
        if q is None:
            q = loss
        q = (1 - visual_smooth) * q + visual_smooth * loss
        q_list.append(q)
        if len(q_list) > 300:
            del q_list[0]
        metric_train = float((lin_model.predict(X_train) == y_train).mean())
        metric_test = float((lin_model.predict(X_test) == y_test).mean())

        tracker.add_metric("loss", "loss" + train_name, loss, i_epoch)
        tracker.add_metric("loss", "Q" + train_name, q, i_epoch)
        tracker.add_metric("accuracy", "test" + train_name, metric_test, i_epoch)
        tracker.add_metric("accuracy", "train" + train_name, metric_train, i_epoch)
        if i_epoch > 300:
            a, b = linear_ab(q_list)
            if abs(a) < 1e-5:
                break
    print(i_epoch)
    if draw_loss:
        tracker.draw_all(max_points=1000)


def train_corr(
    epochs=10_000,
    h=0.001,
    tao=0.01,
    momentum_k=0.01,
    batch_size=2**6,
    visual_smooth=0.01,
):
    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(tao)

    init_correlation(lin_model, X_train, y_train)
    _train(
        X_train,
        X_test,
        y_train,
        y_test,
        lin_model,
        epochs,
        h,
        momentum_k,
        batch_size,
        visual_smooth,
    )

    metric_test = float((lin_model.predict(X_test) == y_test).mean())
    print(metric_test)


def train_speed_sgd(
    epochs=10_000,
    h=None,
    tao=0.01,
    momentum_k=0.1,
    batch_size=2**6,
    visual_smooth=0.01,
):
    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(tao)
    n, m = X_train.shape

    init_random(lin_model, n, m)
    _train(
        X_train,
        X_test,
        y_train,
        y_test,
        lin_model,
        epochs,
        None,
        momentum_k,
        batch_size,
        visual_smooth,
    )

    metric_test = float((lin_model.predict(X_test) == y_test).mean())
    print(metric_test)


def train_multistart(
    epochs=1_000,
    h=0.001,
    tao=0.01,
    momentum_k=0.01,
    batch_size=2**6,
    visual_smooth=0.01,
):
    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(tao)
    n, m = X_train.shape
    for _ in range(10):
        init_random(lin_model, n, m)
        _train(
            X_train,
            X_test,
            y_train,
            y_test,
            lin_model,
            epochs,
            h,
            momentum_k,
            batch_size,
            visual_smooth,
            draw_loss=False,
        )
        metric_test = float((lin_model.predict(X_test) == y_test).mean())
        print(metric_test)


def train_fetch_prob(
    epochs=10_000,
    h=0.001,
    tao=0.01,
    momentum_k=0.01,
    batch_size=2**6,
    visual_smooth=0.01,
):
    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(tao)
    n, m = X_train.shape

    init_random(lin_model, n, m)
    _train(
        X_train,
        X_test,
        y_train,
        y_test,
        lin_model,
        epochs,
        h,
        momentum_k,
        batch_size,
        visual_smooth,
        fetch_prob=True,
    )

    metric_test = float((lin_model.predict(X_test) == y_test).mean())
    print(metric_test)


def main():
    train_speed_sgd()


if __name__ == "__main__":
    main()
