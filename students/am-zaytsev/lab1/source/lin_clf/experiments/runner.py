from dataclasses import dataclass

import numpy as np

from lin_clf.data.utils import fetch_batch
from lin_clf.experiments.config import TrainConfig
from lin_clf.model import Model
from lin_clf.optimizer import SGD
from lin_clf.utils import linear_ab
from lin_clf.visualization import MetricTracker


@dataclass
class TrainResult:
    tracker: MetricTracker
    test_accuracy: float
    epochs_run: int
    model: Model


def train(
    X_train,
    X_test,
    y_train,
    y_test,
    lin_model: Model,
    config: TrainConfig,
) -> TrainResult:
    np.random.seed(config.seed)
    optimizer = SGD(lin_model, config.momentum_k, config.h)

    tracker = MetricTracker()
    q = None
    q_list = list()
    prob = None
    i_epoch = -1
    for i_epoch in range(config.epochs):
        if config.fetch_prob:
            margin = lin_model.predict_margin(lin_model._add_ones(X_train), y_train)
            prob = (1 - margin) ** 2
            prob = (prob / prob.sum())[:, 0]
        mini_batch_x, mini_batch_y = fetch_batch(
            X_train, y_train, config.batch_size, prob
        )

        optimizer.step(mini_batch_x, mini_batch_y)

        loss = float(lin_model.loss(mini_batch_x, mini_batch_y))
        if q is None:
            q = loss
        q = (1 - config.visual_smooth) * q + config.visual_smooth * loss
        q_list.append(q)
        if len(q_list) > 300:
            del q_list[0]
        metric_train = float((lin_model.predict(X_train) == y_train).mean())
        metric_test = float((lin_model.predict(X_test) == y_test).mean())

        tracker.add_metric("loss", "loss" + config.train_name, loss, i_epoch)
        tracker.add_metric("loss", "Q" + config.train_name, q, i_epoch)
        tracker.add_metric("accuracy", "test" + config.train_name, metric_test, i_epoch)
        tracker.add_metric(
            "accuracy", "train" + config.train_name, metric_train, i_epoch
        )
        if i_epoch > 300:
            a, b = linear_ab(q_list)
            if abs(a) < 1e-5:
                break

    test_accuracy = float((lin_model.predict(X_test) == y_test).mean())
    epochs_run = i_epoch + 1
    if config.draw_loss:
        tracker.draw_all(max_points=1000)
    return TrainResult(tracker, test_accuracy, epochs_run, lin_model)
