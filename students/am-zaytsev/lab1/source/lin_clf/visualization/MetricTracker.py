import numpy as np
from plotly import express as px


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

    def draw(self, plot_name, max_points=None, renderer="browser"):
        if plot_name not in self.metrics:
            raise KeyError(f"Unknown plot '{plot_name}'")
        x, y, c = self._series(plot_name, max_points)
        fig = px.line(x=x, y=y, color=c, labels={"x": "epoch", "y": plot_name})
        fig.show(renderer=renderer)
        return fig

    def draw_all(self, max_points=None, renderer="browser"):
        for plot_name in self.metrics:
            self.draw(plot_name, max_points, renderer)
