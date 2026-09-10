from plotly import express as px

from lin_clf.data.utils import load_data, preprocess, draw_corr, fetch_batch
from lin_clf.model.LinearClassifier import LinearClassifier


def main():

    h = 0.001
    tao = 0.01
    momentum_k = 0.01
    batch_size = 32

    df_x, df_y = load_data()

    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(X_train, y_train)
    loss_list = list()
    val_list = list()
    dw_list = list()

    mom_v = None

    for i in range(1000):
        margin = lin_model.predict_margin(lin_model._add_ones(X_train), y_train)
        neg_mag_idx = (margin < 0)[:, 0]
        mini_batch_x, mini_batch_y = fetch_batch(
            X_train[neg_mag_idx], y_train[neg_mag_idx], batch_size
        )

        dL = lin_model.diff(mini_batch_x, mini_batch_y, tao)

        if mom_v is None:
            mom_v = 0
        mom_v = (1 - momentum_k) * mom_v + momentum_k * dL
        lin_model.w = lin_model.w - h * mom_v

        loss = float(lin_model.loss(X_train, y_train))
        val = float((lin_model.predict(X_test) == y_test).mean())
        loss_list.append(loss)
        val_list.append(val)
        dw_list.append(dL.mean())

    plt = px.scatter(val_list)
    plt.show(renderer='browser')
    


if __name__ == "__main__":
    main()
