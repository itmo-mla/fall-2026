from lin_clf.data.utils import load_data, preprocess
from lin_clf.experiments.config import TrainConfig
from lin_clf.experiments.runner import TrainResult, train
from lin_clf.model import LinearClassifier
from lin_clf.model.WeightInit import init_correlation


def train_corr(config: TrainConfig | None = None) -> TrainResult:
    config = config or TrainConfig()
    df_x, df_y = load_data()
    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    lin_model = LinearClassifier(config.tao)
    init_correlation(lin_model, X_train, y_train)
    return train(X_train, X_test, y_train, y_test, lin_model, config)
