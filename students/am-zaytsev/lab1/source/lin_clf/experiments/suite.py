import os
from dataclasses import dataclass, replace

from lin_clf.experiments.config import TrainConfig
from lin_clf.experiments.trainers.corr import train_corr
from lin_clf.experiments.trainers.multistart import MultistartResult, train_multistart
from lin_clf.experiments.runner import TrainResult
from lin_clf.experiments.trainers.speed_sgd import train_speed_sgd


@dataclass
class SuiteResult:
    """Results of the full experiment suite, keyed by run name.

    ``runs`` maps a run name to its ``TrainResult`` and ``aggregates`` maps a
    run name to mean/std statistics for multistart runs.
    """

    runs: dict
    aggregates: dict

    @property
    def accuracies(self):
        return {name: run.test_accuracy for name, run in self.runs.items()}

    @property
    def best_name(self):
        return max(self.runs, key=lambda name: self.runs[name].test_accuracy)


def _best_of(result: MultistartResult) -> TrainResult:
    return max(result.results, key=lambda run: run.test_accuracy)


def run_suite(config: TrainConfig | None = None, restarts: int = 10) -> SuiteResult:
    config = config or TrainConfig()
    base = replace(
        config, momentum_k=0.01, h=0.001, fetch_prob=False, draw_loss=False
    )

    runs = dict()
    aggregates = dict()

    rand = train_multistart(replace(base, fetch_prob=False), restarts=restarts)
    runs["rand"] = _best_of(rand)
    aggregates["rand"] = {"mean": rand.mean, "std": rand.std}

    rand_fetch_prob = train_multistart(
        replace(base, fetch_prob=True), restarts=restarts
    )
    runs["rand_fetch_prob"] = _best_of(rand_fetch_prob)
    aggregates["rand_fetch_prob"] = {
        "mean": rand_fetch_prob.mean,
        "std": rand_fetch_prob.std,
    }

    runs["corr"] = train_corr(replace(base, fetch_prob=False))
    runs["corr_fetch_prob"] = train_corr(replace(base, fetch_prob=True))

    runs["speed_sgd"] = train_speed_sgd(
        replace(base, momentum_k=1.0, h=None, fetch_prob=False)
    )

    return SuiteResult(runs=runs, aggregates=aggregates)


def write_summary(result: SuiteResult, directory) -> str:
    os.makedirs(directory, exist_ok=True)
    lines = list()
    for name, run in result.runs.items():
        line = (
            f"{name}: test_accuracy={run.test_accuracy:.4f} "
            f"epochs_run={run.epochs_run}"
        )
        aggregate = result.aggregates.get(name)
        if aggregate is not None:
            line += f" mean={aggregate['mean']:.4f} std={aggregate['std']:.4f}"
        lines.append(line)
    best = result.best_name
    lines.append(
        f"best: {best} test_accuracy={result.runs[best].test_accuracy:.4f}"
    )
    path = os.path.join(directory, "summary.txt")
    with open(path, "w") as summary:
        summary.write("\n".join(lines) + "\n")
    return path
