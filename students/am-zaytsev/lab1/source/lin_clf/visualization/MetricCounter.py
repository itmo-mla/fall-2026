import numpy as np
from tabulate import tabulate


def _digit2str(d, str_len):
    d = str(d)
    return " " * (str_len - len(d)) + d


class MetricCounter:
    """Computes and prints classification quality metrics from predictions.

    Reports accuracy, recall, precision, f1-score and the confusion matrix.
    """

    def __init__(self, y_pred: np.array, y_true: np.array):
        self.y_pred = y_pred.flatten()
        self.y_true = y_true.flatten()

        self.metric_dict = {
            "acc": self.acc,
            "recall": self.recall,
            "precision": self.precision,
            "f1-score": self.f1_score,
        }

    def conf_mtx(self):
        tp = int(((self.y_true == 1) & (self.y_pred == 1)).sum())
        fp = int(((self.y_true == -1) & (self.y_pred == 1)).sum())
        fn = int(((self.y_true == -1) & (self.y_pred == -1)).sum())
        tn = int(((self.y_true == 1) & (self.y_pred == -1)).sum())

        return [
            [fn, fp],
            [tn, tp],
        ]

    def recall(self):
        [
            [fn, fp],
            [tn, tp],
        ] = self.conf_mtx()
        return tp / (tp + tn)

    def precision(self):
        [
            [fn, fp],
            [tn, tp],
        ] = self.conf_mtx()
        return tp / (tp + fp)

    def f1_score(self):
        pr = self.precision()
        rc = self.recall()
        return 2 * pr * rc / (pr + rc)

    def acc(self):
        [
            [fn, fp],
            [tn, tp],
        ] = self.conf_mtx()
        return (fn + tp) / (fn + fp + tn + tp)

    def _print_cnf_mtx(self):
        [
            [fn, fp],
            [tn, tp],
        ] = self.conf_mtx()
        n = max([len(str(i)) for i in [fn, fp, tn, tp]])
        print(
            f"""=== Confusion Matrix ===
    n   p
f  {_digit2str(fn, n)}  {_digit2str(fp, n)}
t  {_digit2str(tn, n)}  {_digit2str(tp, n)}
"""
        )

    def _dum_metrics(self, dump_list, metic_name_list):
        for k in metic_name_list:
            dump_list.append(self.metric_dict[k]())

    def print_all(self):
        print(f"""=== Model Evaluation ===
Accuracy: {round(self.acc(), 4)}\n""")
        self._print_cnf_mtx()

        metric_list = [["1"], ["-1"]]
        headers = [i for i in self.metric_dict if i != "acc"]
        self._dum_metrics(metric_list[0], headers)
        self.y_pred *= -1
        self.y_true *= -1
        self._dum_metrics(metric_list[1], headers)
        self.y_pred *= -1
        self.y_true *= -1
        headers = ["class"] + headers
        # print(headers)
        # print(metric_list)

        print("=== Classification Report ===")
        print(tabulate(metric_list, headers=headers, floatfmt=".2f"))
