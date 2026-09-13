from dataclasses import dataclass, replace

import numpy as np

from lin_clf.data.utils import load_data, preprocess
from lin_clf.experiments.config import TrainConfig
from lin_clf.experiments.runner import TrainResult, train
from lin_clf.model import LinearClassifier
from lin_clf.model.WeightInit import init_random


@dataclass
class MultistartResult:
    """Aggregated results of several training runs with random initialization."""

    accuracies: list
    mean: float
    std: float
    results: list


def train_multistart(
    config: TrainConfig | None = None, restarts: int = 10
) -> MultistartResult:
    config = config or TrainConfig()
    df_x, df_y = load_data()
    X_train, X_test, y_train, y_test = preprocess(df_x, df_y)

    n, m = X_train.shape
    lin_model = LinearClassifier(config.tao)
    results = list()
    for i in range(restarts):
        run_config = replace(
            config, seed=config.seed + i, draw_loss=False
        )
        np.random.seed(run_config.seed)
        init_random(lin_model, n, m)
        results.append(
            train(X_train, X_test, y_train, y_test, lin_model, run_config)
        )

    accuracies = [result.test_accuracy for result in results]
    return MultistartResult(
        accuracies=accuracies,
        mean=float(np.mean(accuracies)),
        std=float(np.std(accuracies)),
        results=results,
    )
