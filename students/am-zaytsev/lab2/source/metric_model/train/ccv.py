import numpy as np


def get_sorted_m(arr, m):
    return np.quantile(arr, m / (arr.shape[0] - 1), method="nearest")


def get_m_closest_class(item, X, y, m, return_idx=False):
    dist = np.linalg.norm(X - item, axis=1)
    dist_m = get_sorted_m(dist, m)
    m_close_idx = np.argmin(np.abs(dist - dist_m))
    if not return_idx:
        return y[m_close_idx][0]
    return y[m_close_idx][0], m_close_idx


def compact_profile(X: np.array, y: np.array, m: int):
    profile_list = []
    for i in range(X.shape[0]):
        x = X[i]
        y_m = get_m_closest_class(x, X, y, m)
        profile_list.append(int(y_m != y[i, 0]))
    return np.mean(profile_list)


def _color(v):
    colors = ["red", "green", "blue", "brown"]
    return colors[hash(v) % len(colors)]


if __name__ == "__main__":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1. Initialize a subplot grid (1 row, 2 columns)
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Data", " П(m)"))

    n = 1000
    width = 10

    for row_i, dist in enumerate([10, 0, -10]):
        row_i += 1
        x_pos = np.random.uniform(-(width + dist / 2), -(dist / 2), (n // 2, 2))
        y_pos = np.ones_like(x_pos) * -1

        x_neg = np.random.uniform(dist / 2, width + dist / 2, (n // 2, 2))
        y_neg = np.ones_like(x_pos)

        x = np.concat([x_pos, x_neg], 0)
        y = np.concat([y_pos, y_neg], 0)
        x[:, 1] = np.random.uniform(
            -(width + dist / 2), (width + dist / 2), x[:, 1].shape
        )

        fig.add_trace(
            go.Scatter(
                x=x[:, 0],
                y=x[:, 1],
                mode="markers",
                marker=dict(color=list(map(_color, y[:, 0].tolist()))),
            ),
            row=row_i,
            col=1,
        )

        m_list = np.linspace(1, n - 1, 100)
        p_list = [compact_profile(x, y, m) for m in m_list]

        fig.add_trace(
            go.Scatter(x=m_list, mode="markers", y=p_list),
            row=row_i,
            col=2,
        )
        fig.update_xaxes(title_text="M Parameter", row=row_i, col=2)
        fig.update_yaxes(
            title_text="Profile Score", range=[-0.05, 1.05], row=row_i, col=2
        )

    fig.show(renderer="browser")
    # fig.write_image("images/compact_profile_plots.jpg")
