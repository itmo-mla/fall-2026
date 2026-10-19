from dataclasses import replace

from lin_clf.data.utils import load_data, preprocess
from lin_clf.experiments.config import TrainConfig
from lin_clf.experiments.runner import TrainResult, train
from lin_clf.model import LinearClassifier
from lin_clf.model.WeightInit import init_random


def train_fetch_prob(config: TrainConfig | None = None) -> TrainResult:
    config = replace(config or TrainConfig(), fetch_prob=True)
    df_x, df_y = load_data()
    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    n, m = X_train.shape
    lin_model = LinearClassifier(config.tao)
    init_random(lin_model, n, m)
    return train(X_train, X_test, y_train, y_test, lin_model, config)
