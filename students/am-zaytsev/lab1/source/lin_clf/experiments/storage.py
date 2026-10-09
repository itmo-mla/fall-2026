import os

from lin_clf.experiments.runner import TrainResult


def save_result(result: TrainResult, directory, max_points=None, name=None):
    os.makedirs(directory, exist_ok=True)
    result.tracker.save_all(directory, max_points=max_points)
    name = name or os.path.basename(os.path.normpath(directory))
    path = os.path.join(directory, f"{name}_summary.txt")
    with open(path, "w") as summary:
        summary.write(f"test_accuracy {result.test_accuracy}\n")
        summary.write(f"epochs_run {result.epochs_run}\n")
    return path
