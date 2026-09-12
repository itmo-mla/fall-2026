from plotly import express as px
import numpy as np
from lin_clf.data.utils import load_data, preprocess, draw_corr, fetch_batch
from lin_clf.model.LinearClassifier import LinearClassifier


def get_h_star(dL, c1, c2, w):
    return (dL.T @ c1 + dL.T @ c2 @ w) / (dL.T @ c2 @ dL)


def train(h=0.001, tao=0.01, momentum_k=1, batch_size=2**6):

    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(X_train, y_train)
    loss_list = list()
    metric_train_list = list()
    metric_test_list = list()
    dw_list = list()

    mom_v = 0

    for i in range(10000):
        margin = lin_model.predict_margin(lin_model._add_ones(X_train), y_train)
        neg_mag_idx = (np.ones_like(margin) == 1)[:, 0]
        mini_batch_x, mini_batch_y = fetch_batch(
            X_train[neg_mag_idx], y_train[neg_mag_idx], batch_size
        )

        dL, c1, c2 = lin_model.diff(mini_batch_x, mini_batch_y, tao)

        mom_v = (1 - momentum_k) * mom_v + momentum_k * dL
        h = get_h_star(dL, c1, c2, lin_model.w)
        lin_model.w = lin_model.w - h * mom_v

        loss = float(lin_model.loss(X_train, y_train))
        metric_train = float((lin_model.predict(X_train) == y_train).mean())
        metric_test = float((lin_model.predict(X_test) == y_test).mean())
        loss_list.append(loss)
        metric_train_list.append(metric_train)
        metric_test_list.append(metric_test)
        dw_list.append(dL.mean())

    x = list(range(1000))
    y = [
        metric_train_list[int(i)]
        for i in np.linspace(0, len(metric_train_list) - 1, 1000)
    ]
    c = ["train"] * 1000

    x += x
    y += [
        metric_test_list[int(i)]
        for i in np.linspace(0, len(metric_test_list) - 1, 1000)
    ]
    c += ["test"] * 1000

    plt = px.line(x=x, y=y, color=c)
    plt.show(renderer="browser")


def main():
    train()


if __name__ == "__main__":
    main()
