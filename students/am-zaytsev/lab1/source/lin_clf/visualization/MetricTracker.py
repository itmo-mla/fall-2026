import os

import matplotlib.pyplot as plt
import numpy as np


class MetricTracker:
    def __init__(self):
        self.metrics = dict()

    def add_metric(self, plot_name, metric_name, value, epoch):
        self.metrics.setdefault(plot_name, []).append([epoch, value, metric_name])

    def _select(self, rows, max_points):
        if max_points is None or len(rows) <= max_points:
            return rows
        idx = np.linspace(0, len(rows) - 1, max_points).astype(int)
        return [rows[i] for i in idx]

    def _series(self, plot_name, max_points):
        groups = dict()
        for row in self.metrics[plot_name]:
            groups.setdefault(row[2], []).append(row)
        selected = []
        for rows in groups.values():
            selected.extend(self._select(rows, max_points))
        selected.sort(key=lambda row: row[0])
        x = [row[0] for row in selected]
        y = [row[1] for row in selected]
        c = [row[2] for row in selected]
        return x, y, c

    def _figure(self, plot_name, max_points):
        if plot_name not in self.metrics:
            raise KeyError(f"Unknown plot '{plot_name}'")
        x, y, c = self._series(plot_name, max_points)
        groups = dict()
        for xi, yi, ci in zip(x, y, c):
            groups.setdefault(ci, ([], []))
            groups[ci][0].append(xi)
            groups[ci][1].append(yi)
        fig, ax = plt.subplots()
        for name, (gx, gy) in groups.items():
            ax.plot(gx, gy, label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(plot_name)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return fig

    def draw(self, plot_name, max_points=None):
        fig = self._figure(plot_name, max_points)
        plt.show()
        return fig

    def draw_all(self, max_points=None):
        for plot_name in self.metrics:
            self.draw(plot_name, max_points)

    def save(self, plot_name, path, max_points=None, dpi=150):
        fig = self._figure(plot_name, max_points)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    def save_all(self, directory, max_points=None, format="png", dpi=150):
        os.makedirs(directory, exist_ok=True)
        paths = list()
        for plot_name in self.metrics:
            path = os.path.join(directory, f"{plot_name}.{format}")
            self.save(plot_name, path, max_points, dpi)
            paths.append(path)
        return paths
