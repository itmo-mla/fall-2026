import numpy as np
from lin_clf.data.utils import load_data, preprocess, draw_corr, fetch_batch
from lin_clf.model import LinearClassifier
from lin_clf.optimizer import SGD
from lin_clf.visualization import MetricTracker


def train(
    epochs=10_000,
    h=0.001,
    tao=0.01,
    momentum_k=0.01,
    batch_size=2**6,
    visual_smooth=0.01,
):

    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    n, m = X_train.shape
    lin_model = LinearClassifier(tao)
    lin_model.init_weights(n, m)
    optimizer = SGD(lin_model, momentum_k, h)

    tracker = MetricTracker()
    q = None

    for i in range(epochs):
        margin = lin_model.predict_margin(lin_model._add_ones(X_train), y_train)
        neg_mag_idx = (np.ones_like(margin) == 1)[:, 0]
        mini_batch_x, mini_batch_y = fetch_batch(
            X_train[neg_mag_idx], y_train[neg_mag_idx], batch_size
        )

        optimizer.step(mini_batch_x, mini_batch_y)

        loss = float(lin_model.loss(mini_batch_x, mini_batch_y))
        if q is None:
            q = loss
        q = (1 - visual_smooth) * q + visual_smooth * loss
        metric_train = float((lin_model.predict(X_train) == y_train).mean())
        metric_test = float((lin_model.predict(X_test) == y_test).mean())

        tracker.add_metric("loss", "loss", loss, i)
        tracker.add_metric("loss", "Q", q, i)
        tracker.add_metric("accuracy", "test", metric_test, i)
        tracker.add_metric("accuracy", "train", metric_train, i)

    tracker.draw_all(max_points=1000)


def main():
    train()


if __name__ == "__main__":
    main()
